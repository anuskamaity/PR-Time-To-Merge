import requests, os, time
import pandas as pd

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def fetch_prs_list(owner, repo, max_pages=3):
    """Fetches a list of closed PRs (summary level)."""
    all_prs = []
    for page in range(1, max_pages + 1):
        print(f"  Fetching page {page} of PR list for {owner}/{repo}...")
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=closed&per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"  Error: {response.status_code} - {response.text}")
            break
            
        data = response.json()
        if not data:
            break
        all_prs.extend(data)
        time.sleep(0.5) # Avoid hitting secondary rate limits
    return all_prs

def fetch_pr_details(owner, repo, pr_number):
    """Fetches detailed PR info (additions, deletions, changed_files, commits)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    return {}

def collect_repo_data(owner, repo, max_prs=150):
    """Orchestrates fetching list + details for a specific repository."""
    print(f"\nStarting ingestion for {owner}/{repo}...")
    
    # 1. Get the list of closed PRs
    raw_list = fetch_prs_list(owner, repo)
    
    # 2. Filter for only merged PRs (ignore closed-but-unmerged)
    merged_prs = [pr for pr in raw_list if pr.get("merged_at") is not None]
    print(f"  Found {len(merged_prs)} merged PRs. Fetching details for top {max_prs}...")

    detailed_rows = []
    # 3. Fetch deep details for each PR to get 'additions', 'deletions', etc.
    for i, pr in enumerate(merged_prs[:max_prs]):
        pr_num = pr["number"]
        if i % 10 == 0:
            print(f"    Processing PR {i}/{max_prs}...")
            
        details = fetch_pr_details(owner, repo, pr_num)
        
        # Calculate target variable: Hours to Merge
        # created_at = pd.to_datetime(pr["created_at"])
        # merged_at = pd.to_datetime(pr["merged_at"])
        # hours_to_merge = (merged_at - created_at).total_seconds() / 3600

        # Feature Selection: Focus on things available at/shortly after creation
        detailed_rows.append({
            "repo": f"{owner}/{repo}",
            "pr_number": pr_num,
            "created_at": pr["created_at"],
            "merged_at": pr["merged_at"],
            "title_len": len(pr["title"]) if pr["title"] else 0,
            "body_len": len(pr["body"]) if pr["body"] else 0,
            "author_assoc": pr["author_association"],
            "num_labels": len(pr.get("labels", [])),
            "is_draft": pr.get("draft", False),
            "additions": details.get("additions", 0),
            "deletions": details.get("deletions", 0),
            "changed_files": details.get("changed_files", 0),
            "num_commits": details.get("commits", 0),
            "total_comments": details.get("comments", 0) + details.get("review_comments", 0)
        })
        time.sleep(0.2) # To avoid triggering rate limits

    return pd.DataFrame(detailed_rows)

if __name__ == "__main__":
    if not TOKEN:
        print("ERROR: Please set the GITHUB_TOKEN environment variable.")
    else:
        # Create data directory
        os.makedirs("data", exist_ok=True)

        # Repositories suggested in the assignment
        repos = [
            ("microsoft", "vscode"),
            ("excalidraw", "excalidraw")
        ]

        all_dfs = []
        for owner, repo in repos:
            repo_df = collect_repo_data(owner, repo, max_prs=200)
            all_dfs.append(repo_df)

        # Combine, Save and Report
        final_df = pd.concat(all_dfs, ignore_index=True)
        output_path = "data/prs_merged_cleaned.csv"
        final_df.to_csv(output_path, index=False)
        
        print(f"Total PRs collected: {len(final_df)}")
        print(f"File saved to: {output_path}")