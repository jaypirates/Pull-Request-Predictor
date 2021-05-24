import json
import os
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import requests

load_dotenv(find_dotenv())

# This is the JSON array that is used to collect the data and finally dump it to the output file
final_object = []

# Dictionaries used to convert categorical data into numerical data
pr_status_checks_dict = {"success": 1, "pending": 2, "error": 3, "failure": 3}
pr_user_type_dict = {"OWNER": 1, "MEMBER": 2, "COLLABORATOR": 3, "CONTRIBUTOR": 4, "FIRST_TIME_CONTRIBUTOR": 5, "NONE": 6, "FIRST_TIMER": 7}

# vars holding url params
owner = "sequelize"
repos = "sequelize"

def append_data(json_response, output_object):
    for pr in json_response:
        # Calling the status checks API endpoint
        status_check = requests.get("https://api.github.com/repos/%s/%s/commits/%s/status" % (owner, repos, pr["head"]["sha"]), headers={'Authorization': 'token %s' % (os.getenv("access_token"))})
        status_check = status_check.json()

        # Calling the repo commits API endpoint
        user_contr = requests.get("https://api.github.com/repos/%s/%s/commits?author=%s&per_page=100" % (owner, repos, pr["head"]["user"]["login"]), headers={'Authorization': 'token %s' % (os.getenv("access_token"))})
        user_contr = user_contr.json()

        comment_count = requests.get("https://api.github.com/repos/%s/%s/issues/%s/comments?per_page=100" % (owner, repos, pr["number"]), headers={'Authorization': 'token %s' % (os.getenv("access_token"))})
        comment_count = comment_count.json()

        closed_at = pr["closed_at"]
        created_at = pr["created_at"]
        # Parsing string into datetime object
        created_at_timeobject = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        closed_at_timeobject = datetime.strptime(closed_at, "%Y-%m-%dT%H:%M:%SZ")
        time = closed_at_timeobject - created_at_timeobject

        # Converting merged_at categorical data to numerical
        if pr["merged_at"] is None:
            pr["merged_at"] = 0
        else:
            pr["merged_at"] = 1

        pr_number = pr["number"]
        # Using dict to get pr_status_check and pr_user_type numerical values
        pr_status_check = pr_status_checks_dict.get(status_check["state"])
        pr_user_type = pr_user_type_dict.get(pr["author_association"])
        # pr_user_contr is the number of contributions
        pr_user_contr = len(user_contr)
        # pr_comment_count is the number of comments
        pr_comment_count = len(comment_count)
        # Converting time duration to seconds
        pr_time_duration = time.total_seconds()
        pr_is_merged = pr["merged_at"]

        print("PR Number = ", pr_number)
        print("PR Status Check = ", pr_status_check)
        print("PR User Type = ", pr_user_type)
        print("PR User Contribution History = ", pr_user_contr)
        print("PR Conversation History (Comment Count) = ", pr_comment_count)
        print("PR Time Duration", pr_time_duration)
        print("PR Is Merged = ", pr_is_merged, "\n")

        # Appending the PR data to the JSON array
        output_object.append({"pr_number": pr_number, "pr_status_check": pr_status_check, "pr_user_type": pr_user_type, "pr_user_contr": pr_user_contr, "pr_comment_count": pr_comment_count, "pr_time_duration": pr_time_duration, "pr_is_merged": pr_is_merged})


for page_no in range(1, 6):
    print("\npage_no = ", page_no, "\n")
    # API call to the Pull Request endpoint
    response = requests.get("https://api.github.com/repos/%s/%s/pulls?state=closed&page=1&per_page=20&page=%i" % (owner, repos, page_no), headers={'Authorization': 'token %s' % (os.getenv("access_token"))})
    # Converting API response to JSON
    api_response = response.json()
    append_data(api_response, final_object)

# Dumping the JSON array to the JSON file
with open("ouput.json", "w", encoding="utf-8") as f:
    json.dump(final_object, f, ensure_ascii=False, indent=4)

print(len(final_object))
