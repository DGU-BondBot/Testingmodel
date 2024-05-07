import os
import requests

# Create the 'files' directory if it doesn't exist
os.makedirs("../files", exist_ok=True)

# URL of the webpage to download
url = "https://namu.wiki/w/%EC%A7%91%EB%8B%A8%EC%A3%BC%EC%9D%98"

# Send a GET request to fetch the webpage content
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Save the webpage content to a local file
    with open("../files/wiki.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Webpage downloaded successfully and saved as 'wiki.txt'")
else:
    print("Failed to download the webpage. Status code:", response.status_code)
