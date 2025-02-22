import requests
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict

def get_bugsnag_errors(api_key: str, project_id: str, days_back: int = 7) -> List[Dict]:
    """Fetch errors from Bugsnag API and return formatted data"""
    since = datetime.now() - timedelta(days=days_back)
    
    response = requests.get(
        f'https://api.bugsnag.com/projects/{project_id}/errors',
        headers={
            'Authorization': f'token {api_key}',
            'X-Version': '2',
            'Content-Type': 'application/json'
        },
        params={
            'filters[since][0]': since.isoformat(),
            'filters[sort][0]': '-last_seen',
            'per_page': 50
        }
    )
    response.raise_for_status()
    return response.json()

def format_errors_for_llm(errors: List[Dict]) -> str:
    """Format errors in a clean, readable way for pasting into Gemini"""
    formatted = []
    
    for error in errors:
        error_info = {
            'error_class': error['error_class'],
            'message': error['message'],
            'events_count': error['events'],
            'first_seen': error['first_seen'],
            'last_seen': error['last_seen'],
            'context': error.get('context', 'N/A'),
            'release_stage': error.get('release_stage', 'N/A'),
            'stack_trace': error['stacktrace'][0] if error.get('stacktrace') else 'N/A'
        }
        formatted.append(error_info)
    
    return json.dumps(formatted, indent=2)

if __name__ == "__main__":
    # Get credentials from environment variables
    api_key = os.getenv('BUGSNAG_API_KEY')
    project_id = os.getenv('BUGSNAG_PROJECT_ID')
    
    if not api_key or not project_id:
        print("Please set BUGSNAG_API_KEY and BUGSNAG_PROJECT_ID environment variables")
        exit(1)
    
    try:
        # Fetch and format errors
        errors = get_bugsnag_errors(api_key, project_id)
        formatted_output = format_errors_for_llm(errors)
        
        # Print to console for easy copying
        print("\nBugsnag Errors (last 7 days):")
        print("============================")
        print(formatted_output)
        
    except Exception as e:
        print(f"Error fetching Bugsnag data: {e}")
