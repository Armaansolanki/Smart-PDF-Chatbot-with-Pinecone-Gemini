from google_auth_oauthlib.flow import InstalledAppFlow
import os

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']  # ✅ This is the correct scope

def generate_token():
    creds = None
    if os.path.exists("token.json"):
        print("✅ token.json already exists.")
        return

    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
    creds = flow.run_local_server(port=0)

    with open("token.json", "w") as token_file:
        token_file.write(creds.to_json())
        print("✅ token.json has been generated!")

generate_token()

