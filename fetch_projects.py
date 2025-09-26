import requests
import os
import json
import csv
from datetime import datetime, timedelta

GITHUB_API = "https://api.github.com/search/repositories"
HEADERS = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}
TOPICS = ["ai", "ai-agent", "llm"]  # Fetch separately for each topic
PER_PAGE = 50
PAGES = 10
MIN_STARS = 10000
OUTPUT_FILE = "README.md"  # Output file name changed to README.md
EXCLUDE_FILE = "excluded-repos.txt"  # File containing repos to exclude
CSV_FILE = "repos.csv"  # CSV file to store repository data

# OpenAI API configuration
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json"
}


def fetch_repositories(topic):
    repos = []
    one_year_ago = datetime.now() - timedelta(days=365)  # Calculate the date one year ago
    for page in range(1, PAGES + 1):
        params = {
            "q": f"topic:{topic} stars:>={MIN_STARS} pushed:>={one_year_ago.strftime('%Y-%m-%d')}",
            "sort": "stars",  # Sort by stars
            "order": "desc",   # Descending order
            "per_page": PER_PAGE,
            "page": page
        }
        response = requests.get(GITHUB_API, headers=HEADERS, params=params)
        repos_curr = response.json().get("items", [])
        if len(repos_curr) == 0:
            break
        repos.extend(repos_curr)
        print(f"Repos added {len(repos_curr)} of {len(repos)} at topic [{topic}] page [{page}]")
    
    return repos


def format_stars(stars):
    """Format the number of stars to be in k or m units."""
    if stars >= 1_000_000:
        return f"{stars / 1_000_000:.1f}m"  # Format to 'm' for millions
    elif stars >= 1_000:
        return f"{stars / 1_000:.1f}k"  # Format to 'k' for thousands
    else:
        return str(stars)  # No formatting for less than 1000

def load_excluded():
    """Load excluded repositories from excluded-repos.txt (owner/repo format)."""
    excluded_repos = set()
    with open(EXCLUDE_FILE, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                excluded_repos.add(line)
    return excluded_repos

def load_extra_repos():
    """Load additional repositories from extra-repos.txt."""
    extra_repos = []
    with open("extra-repos.txt", "r") as file:
        for line in file:
            owner_repo = line.strip()
            if owner_repo:
                owner, repo = owner_repo.split("/")
                extra_repos.append({"owner": owner, "name": repo})
    return extra_repos

def read_existing_repos_from_csv():
    """Read existing repositories from CSV file."""
    existing_repos = {}
    existing_repo_ids = set()
    
    if not os.path.exists(CSV_FILE):
        return existing_repos, existing_repo_ids
    
    try:
        with open(CSV_FILE, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                repo_id = int(row['id'])
                existing_repo_ids.add(repo_id)
                existing_repos[repo_id] = {
                    'id': repo_id,
                    'owner': row['owner'],
                    'name': row['name'],
                    'stars': int(row['stars']),
                    'url': row['url'],
                    'business_model': row.get('business_model', '')
                }
        print(f"Loaded {len(existing_repos)} repositories from CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}, set()
    
    return existing_repos, existing_repo_ids

def fetch_extra_repo_details(extra_repos):
    """Fetch details for repositories listed in extra-repos.txt."""
    detailed_repos = []
    for repo in extra_repos:
        url = f"https://api.github.com/repos/{repo['owner']}/{repo['name']}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            detailed_repos.append(response.json())
        else:
            print(f"Failed to fetch details for {repo['owner']}/{repo['name']}")
    return detailed_repos

def generate_business_model(repository):
    prompt = f"""
You are an AI business consultant. Describe following repository in one sentence (around 25 words) how it can help me make money. Highlight keywords in bold (e.g., services, SaaS, automation, chatbot, NFT, templates) and clearly mention the monetization approaches such as subscriptions, project-based fees, hosting services, selling templates, or consulting and so on.

- Repository: {repository['name']}
- Description: {repository['description'] or 'No description'}
- URL: {repository['html_url']}
- Stars: {repository['stargazers_count']}

Return only the 25-word business analysis as plain text (no JSON, no formatting, no extra explanation).
"""

    payload = {
        "model": "gpt-5-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a business analyst expert in AI monetization. Provide concise, actionable business insights."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
    }
    
    response = requests.post(OPENAI_API_URL, headers=OPENAI_HEADERS, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    result = response.json()
    business_model = result['choices'][0]['message']['content'].strip()
    
    print(f"Generated business model for {repository['name']}")
    return business_model


def convert_csv_to_readme():
    """Read all data from CSV and convert to README.md format, sorted by stars."""
    # Read existing repos from CSV (including any new ones we just added)
    existing_repos, _ = read_existing_repos_from_csv()
    
    if not existing_repos:
        print("No repositories found in CSV")
        return
    
    # Convert to list and sort by stars in descending order
    repos_list = list(existing_repos.values())
    repos_list.sort(key=lambda x: x['stars'], reverse=True)
    
    # Save to markdown file
    with open(OUTPUT_FILE, "w") as file:
        file.write("# Make Money With AI\n\n")
        file.write("**Make Money With AI** is a curated list of AI tools and projects that help you turn open-source into income.\n\n")
        
        for index, repo_data in enumerate(repos_list, start=1):
            stars = format_stars(repo_data['stars'])
            business_model = repo_data.get('business_model', '')
            file.write(f"{index}. **[{repo_data['name']}]({repo_data['url']})** | ☆{stars} | {business_model}\n")
    
    print(f"Converted {len(repos_list)} repositories from CSV to README.md")

def save_repos_to_csv(repos_dict):
    """Save all repositories to CSV file."""
    if not repos_dict:
        print("No repositories to save to CSV")
        return
    
    # Convert to list and sort by stars for CSV consistency
    repos_list = list(repos_dict.values())
    repos_list.sort(key=lambda x: x['stars'], reverse=True)
    
    # Save to CSV file
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'owner', 'name', 'stars', 'url', 'business_model'])
        
        for repo_data in repos_list:
            writer.writerow([
                repo_data['id'],
                repo_data['owner'],
                repo_data['name'],
                repo_data['stars'],
                repo_data['url'],
                repo_data.get('business_model', '')
            ])
    
    print(f"Saved {len(repos_list)} repositories to CSV file")


if __name__ == "__main__":
    if not os.getenv('GITHUB_TOKEN'):
        raise ValueError("GITHUB_TOKEN is not set")
    
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY is not set")

    # 1. Read existing repos from CSV file
    print("Reading existing repositories from CSV...")
    existing_repos, existing_repo_ids = read_existing_repos_from_csv()
    
    # 2. Fetch all repos from GitHub API
    all_repos = []
    excluded_repos = load_excluded()
    for topic in TOPICS:
        all_repos.extend(fetch_repositories(topic))
    
    # Add extra repositories
    extra_repos = load_extra_repos()
    all_repos.extend(fetch_extra_repo_details(extra_repos))
    print("Loaded extra repositories")
    
    # 3. Process all repos and generate business models for new ones
    seen = set()
    unique_repos = []
    
    for repo in all_repos:
        if repo["id"] not in seen and repo["name"] not in excluded_repos:
            seen.add(repo["id"])
            unique_repos.append(repo)
            
            # Check if this repo is new (not in existing CSV) and generate business model
            if repo["id"] not in existing_repo_ids:
                business_model = generate_business_model(repo)
                
                # Add to existing repos with business model
                existing_repos[repo["id"]] = {
                    'id': repo["id"],
                    'owner': repo['owner']['login'],
                    'name': repo['name'],
                    'stars': repo['stargazers_count'],
                    'url': repo['html_url'],
                    'business_model': business_model
                }
    
    print(f"Total unique repos: {len(unique_repos)}")
    
    # 4. Save all repositories (existing + new) to CSV
    save_repos_to_csv(existing_repos)
    
    # 5. Read from CSV and convert to README.md
    convert_csv_to_readme()
