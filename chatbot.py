import os
import logging
import json
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "Medical_Coder"

# Initialize FastAPI app
app = FastAPI(title="ICD-10 Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the ICD-10 index data
def load_icd10_data():
    icd10_index_path = os.getenv("ICD10_INDEX_PATH")
    with open(icd10_index_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load the ICD-10 additional data
def load_icd10_additional_data():
    icd10_updated_path = os.getenv("ICD10_UPDATED_PATH")
    try:
        with open(icd10_updated_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading ICD-10 additional data: {e}")
        return []

# Global variables to store the data
icd10_data = load_icd10_data()
icd10_additional_data = load_icd10_additional_data()

# Import the optimized search module
from icd_search import router as search_router

# Register the search router
app.include_router(search_router, prefix="/api")

# Helper function to find a term in children
def find_term_in_children(children, path, parent_path=""):
    """Recursively search for a term in children based on path"""
    for child in children:
        child_path = child.get('path', '')
        
        # Handle both string and object paths
        if isinstance(child_path, dict):
            path_text = child_path.get('_', '')
            path_nemod = child_path.get('nemod', '')
            full_path = f"{path_text} {path_nemod}".strip()
        else:
            full_path = child_path
            
        if full_path == path:
            return child
            
        # If this child has children, search them too
        if 'children' in child:
            result = find_term_in_children(child['children'], path, full_path)
            if result:
                return result
                
        # Also check terms array if it exists
        if 'terms' in child:
            result = find_term_in_children(child['terms'], path, full_path)
            if result:
                return result
                
    return None

def search_icd10(query: str, limit: int = 20):
    """Search for ICD-10 codes based on the query using the JSON index"""
    try:
        query = query.lower()
        if not query:
            return []
            
        # Search for terms that match the query at any level
        results = []

        for letter in icd10_data['letters']:
            for term in letter['terms']:
                # Process the term title
                term_title = term.get('title', '')
                display_title = ''
                
                # Handle both string and object titles
                if isinstance(term_title, dict):
                    term_text = term_title.get('_', '')
                    nemod = term_title.get('nemod', '')
                    display_title = f"{term_text} {nemod}".strip()
                else:
                    display_title = term_title
                
                # Check if the term has a path, if not use title as path
                term_path = term.get('path', term_title)
                path_string = ''
                
                # Convert path object to string if needed
                if isinstance(term_path, dict):
                    path_text = term_path.get('_', '')
                    path_nemod = term_path.get('nemod', '')
                    path_string = f"{path_text} {path_nemod}".strip()
                else:
                    path_string = str(term_path)
                
                # Check if query matches the title
                if query in display_title.lower():
                    code = term.get('code', '')
                    see = term.get('see', '')
                    see_also = term.get('seeAlso', '')
                    
                    results.append({
                        'title': display_title,
                        'path': path_string,
                        'letter': letter['title'],
                        'code': code,
                        'see': see,
                        'seeAlso': see_also,
                        'has_children': 'terms' in term or 'children' in term,
                        'original_term': term
                    })
                    
                    # Limit results to prevent overwhelming the UI
                    if len(results) >= limit:
                        break
                
                # Also search in child terms if they exist
                if 'terms' in term and len(results) < limit:
                    child_results = search_in_child_terms(term['terms'], query, letter['title'], path_string, limit - len(results))
                    results.extend(child_results)
                
                # Also search in children if they exist (alternative structure)
                if 'children' in term and len(results) < limit:
                    child_results = search_in_child_terms(term['children'], query, letter['title'], path_string, limit - len(results))
                    results.extend(child_results)
                
                if len(results) >= limit:
                    break
            
            if len(results) >= limit:
                break

        return results
    except Exception as e:
        logger.error(f"Error in search_icd10: {e}")
        return []

def search_in_child_terms(terms, query, letter, parent_path, limit):
    """Helper function to search in child terms"""
    results = []
    
    for term in terms:
        if len(results) >= limit:
            break
            
        # Process the term title
        term_title = term.get('title', '')
        display_title = ''
        
        # Handle both string and object titles
        if isinstance(term_title, dict):
            term_text = term_title.get('_', '')
            nemod = term_title.get('nemod', '')
            display_title = f"{term_text} {nemod}".strip()
        else:
            display_title = term_title
        
        # Check if the term has a path, if not construct from parent path
        term_path = term.get('path', '')
        path_string = ''
        
        # Convert path object to string if needed
        if isinstance(term_path, dict):
            path_text = term_path.get('_', '')
            path_nemod = term_path.get('nemod', '')
            path_string = f"{path_text} {path_nemod}".strip()
        elif term_path:
            path_string = str(term_path)
        else:
            # If no path, construct from parent and title
            path_string = f"{parent_path} > {display_title}"
        
        # Check if query matches the title
        if query in display_title.lower():
            code = term.get('code', '')
            see = term.get('see', '')
            see_also = term.get('seeAlso', '')
            
            results.append({
                'title': display_title,
                'path': path_string,
                'letter': letter,
                'code': code,
                'see': see,
                'seeAlso': see_also,
                'has_children': 'terms' in term or 'children' in term,
                'original_term': term,
                'parent_path': parent_path
            })
        
        # Recursively search in child terms if they exist
        if 'terms' in term and len(results) < limit:
            child_results = search_in_child_terms(term['terms'], query, letter, path_string, limit - len(results))
            results.extend(child_results)
        
        # Recursively search in children if they exist (alternative structure)
        if 'children' in term and len(results) < limit:
            child_results = search_in_child_terms(term['children'], query, letter, path_string, limit - len(results))
            results.extend(child_results)
    
    return results

def get_term_details(letter: str, path: str, parent_path: str = ''):
    """Get detailed information for a specific ICD-10 term"""
    try:
        if not letter or not path:
            return {"error": "Missing parameters"}
            
        # Special case for Diabetes term
        if letter == "D" and ("Diabetes" in path or path == "{'_': 'Diabetes, diabetic', 'nemod': '(mellitus) (sugar)'}"):
            # Find the Diabetes term directly
            for letter_obj in icd10_data['letters']:
                if letter_obj['title'] == "D":
                    for term in letter_obj['terms']:
                        title = term['title']
                        if isinstance(title, dict) and title.get('_') == 'Diabetes, diabetic':
                            return {
                                'term': term,
                                'has_children': 'children' in term and len(term['children']) > 0
                            }

        # Special case for "with" term under Diabetes
        if path == "[object Object] > with" and letter == "D":
            # Check if the previous term was Diabetes
            if parent_path and ("Diabetes" in parent_path or parent_path == "{'_': 'Diabetes, diabetic', 'nemod': '(mellitus) (sugar)'}"):
                # Find the Diabetes term first
                diabetes_term = None
                for letter_obj in icd10_data['letters']:
                    if letter_obj['title'] == "D":
                        for term in letter_obj['terms']:
                            title = term['title']
                            if isinstance(title, dict) and title.get('_') == 'Diabetes, diabetic':
                                diabetes_term = term
                                break

                if diabetes_term and 'children' in diabetes_term:
                    # Find the "with" child
                    for child in diabetes_term['children']:
                        if child['title'] == 'with':
                            return {
                                'term': child,
                                'has_children': 'children' in child and len(child['children']) > 0
                            }
            else:
                # If no parent_path or not from Diabetes, find the Diabetes term first
                diabetes_term = None
                for letter_obj in icd10_data['letters']:
                    if letter_obj['title'] == "D":
                        for term in letter_obj['terms']:
                            title = term['title']
                            if isinstance(title, dict) and title.get('_') == 'Diabetes, diabetic':
                                diabetes_term = term
                                break

                if diabetes_term and 'children' in diabetes_term:
                    # Find the "with" child
                    for child in diabetes_term['children']:
                        if child['title'] == 'with':
                            return {
                                'term': child,
                                'has_children': 'children' in child and len(child['children']) > 0
                            }

        # Regular path handling
        for letter_obj in icd10_data['letters']:
            if letter_obj['title'] == letter:
                for term in letter_obj['terms']:
                    term_path = term['path']

                    # Handle both string and object paths
                    if isinstance(term_path, dict):
                        term_path_text = term_path.get('_', '')
                        term_path_nemod = term_path.get('nemod', '')
                        full_path = f"{term_path_text} {term_path_nemod}".strip()
                    else:
                        full_path = term_path

                    if full_path == path:
                        # Return the term and its children if any
                        return {
                            'term': term,
                            'has_children': 'children' in term and len(term['children']) > 0
                        }

                    # If the term has children, check them recursively
                    if 'children' in term:
                        child_term = find_term_in_children(term['children'], path, parent_path)
                        if child_term:
                            return {
                                'term': child_term,
                                'has_children': 'children' in child_term and len(child_term['children']) > 0
                            }

        return {"error": "Term not found"}
    except Exception as e:
        logger.error(f"Error in get_term_details: {e}")
        return {"error": str(e)}

@app.get('/api/search')
def search(request: Request):
    query = request.query_params.get('query', '').lower()

    if not query:
        return json.dumps([])

    # Search for level 0 terms that match the query
    results = []

    for letter in icd10_data['letters']:
        for term in letter['terms']:
            if term.get('level', 0) == 0:  # Safely access level with default 0
                term_title = term.get('title', '')
                # Safely access path, fallback to title if path doesn't exist
                term_path = term.get('path', term_title)

                # Handle both string and object titles
                if isinstance(term_title, dict):
                    term_text = term_title.get('_', '')
                    nemod = term_title.get('nemod', '')
                    display_title = f"{term_text} {nemod}".strip()
                else:
                    display_title = term_title

                # Convert path object to string if needed
                if isinstance(term_path, dict):
                    path_text = term_path.get('_', '')
                    path_nemod = term_path.get('nemod', '')
                    path_string = f"{path_text} {path_nemod}".strip()
                else:
                    path_string = str(term_path) if term_path else display_title  # Ensure we have a string

                if query in display_title.lower():
                    results.append({
                        'title': display_title,
                        'path': path_string,
                        'letter': letter['title'],
                        'original_term': term
                    })

                    # Limit results to prevent overwhelming the UI
                    if len(results) >= 20:
                        break

        if len(results) >= 20:
            break

    return json.dumps(results)

@app.get('/api/term')
def get_term(request: Request):
    letter = request.query_params.get('letter')
    path = request.query_params.get('path')
    parent_path = request.query_params.get('parent_path', '')

    if not letter or not path:
        return JSONResponse(content={'error': 'Missing parameters'}, status_code=400)

    result = get_term_details(letter, path, parent_path)
    return JSONResponse(content=result)

def find_children_for_path(letter: str, path: str, parent_path: str = ''):
    """Find children for a specific path"""
    try:
        # Get the term details first
        term_details = get_term_details(letter, path, parent_path)
        
        if 'error' in term_details:
            return {'error': term_details['error']}
            
        term = term_details.get('term', {})
        
        # Check if the term has children
        children = []
        
        # Check both 'children' and 'terms' fields
        if 'children' in term:
            children = term['children']
        elif 'terms' in term:
            children = term['terms']
        
        # Format the children for response
        formatted_children = []
        for child in children:
            child_title = child.get('title', '')
            display_title = ''
            
            # Handle both string and object titles
            if isinstance(child_title, dict):
                title_text = child_title.get('_', '')
                nemod = child_title.get('nemod', '')
                display_title = f"{title_text} {nemod}".strip()
            else:
                display_title = child_title
                
            # Get child path
            child_path = child.get('path', '')
            path_string = ''
            
            # Handle both string and object paths
            if isinstance(child_path, dict):
                path_text = child_path.get('_', '')
                path_nemod = child_path.get('nemod', '')
                path_string = f"{path_text} {path_nemod}".strip()
            elif child_path:
                path_string = str(child_path)
            else:
                # If no path, construct from parent and title
                path_string = f"{path} > {display_title}"
                
            # Add to formatted children
            formatted_children.append({
                'title': display_title,
                'path': path_string,
                'code': child.get('code', ''),
                'see': child.get('see', ''),
                'seeAlso': child.get('seeAlso', ''),
                'has_children': ('children' in child and len(child['children']) > 0) or 
                               ('terms' in child and len(child['terms']) > 0)
            })
            
        return {'children': formatted_children}
    except Exception as e:
        logger.error(f"Error in find_children_for_path: {e}")
        return {'error': str(e)}

@app.get('/api/term-children')
def get_term_children(request: Request):
    """Get children/subterms for a specific term path"""
    letter = request.query_params.get('letter')
    path = request.query_params.get('path')
    parent_path = request.query_params.get('parent_path', '')

    if not letter or not path:
        return JSONResponse(content={'error': 'Missing parameters'}, status_code=400)
        
    result = find_children_for_path(letter, path, parent_path)
    return JSONResponse(content=result)

@app.get("/api/code/{code}")
def get_code_details(code: str):
    """Get detailed information for a specific ICD-10 code"""
    try:
        # Search through all letters and terms to find the code
        for letter in icd10_data['letters']:
            for term in letter['terms']:
                # Check if this term has the code
                if term.get('code') == code:
                    return {
                        'term': term,
                        'has_children': 'children' in term or 'terms' in term
                    }
                
                # Check in children recursively
                if 'children' in term:
                    found_term = find_code_in_children(term['children'], code)
                    if found_term:
                        return {
                            'term': found_term,
                            'has_children': 'children' in found_term or 'terms' in found_term
                        }
                
                # Check in terms array if it exists
                if 'terms' in term:
                    found_term = find_code_in_children(term['terms'], code)
                    if found_term:
                        return {
                            'term': found_term,
                            'has_children': 'children' in found_term or 'terms' in found_term
                        }
        
        # If we get here, the code wasn't found
        return {"error": f"Code {code} not found"}
    except Exception as e:
        logger.error(f"Error in get_code_details: {e}")
        return {"error": str(e)}

def find_code_in_children(children, code):
    """Recursively search for a code in children"""
    for child in children:
        # Check if this child has the code
        if child.get('code') == code:
            return child
        
        # Check in this child's children
        if 'children' in child:
            found = find_code_in_children(child['children'], code)
            if found:
                return found
        
        # Check in terms array if it exists
        if 'terms' in child:
            found = find_code_in_children(child['terms'], code)
            if found:
                return found
    
    return None

@app.get("/api/code-children")
def get_code_children(request: Request):
    """Get children/subterms for a specific ICD-10 code"""
    code = request.query_params.get('code')
    
    if not code:
        return JSONResponse(content={'error': 'Missing code parameter'}, status_code=400)
    
    try:
        # First find the term with this code
        term = None
        for letter in icd10_data['letters']:
            for t in letter['terms']:
                if t.get('code') == code:
                    term = t
                    break
                
                # Check in children recursively
                if not term and 'children' in t:
                    term = find_code_in_children(t['children'], code)
                    if term:
                        break
                
                # Check in terms array if it exists
                if not term and 'terms' in t:
                    term = find_code_in_children(t['terms'], code)
                    if term:
                        break
            
            if term:
                break
        
        if not term:
            return JSONResponse(content={"error": f"Code {code} not found"}, status_code=404)
        
        # Return children or terms if they exist
        if 'children' in term and term['children']:
            return JSONResponse(content=term['children'])
        elif 'terms' in term and term['terms']:
            return JSONResponse(content=term['terms'])
        else:
            return JSONResponse(content=[])
    except Exception as e:
        logger.error(f"Error in get_code_children: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Legacy endpoint kept for backward compatibility
@app.get("/api/lookup_code/{code}")
async def lookup_code(code: str, limit: int = 20):
    """Legacy API endpoint to lookup additional data for a specific ICD-10 code
    This is kept for backward compatibility. Use /api/search/{code} for better performance.
    """
    try:
        # Redirect to the optimized search endpoint
        from icd_search import search_code
        return await search_code(code=code, limit=limit)
    except Exception as e:
        logger.error(f"Error looking up code {code}: {e}")
        return JSONResponse(content={"error": f"Failed to lookupCode: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)