import dropbox
import requests

class DropboxAPI:
    def __init__(self, mk1):
        """
            Step 1 : https://developers.dropbox.com/oauth-guide
            Step 2 : https://www.dropbox.com/oauth2/authorize?client_id=<CLIENT_ID>&fromDws=True&response_type=code&token_access_type=offline
            Step 3 : Execute the function get_refresh_token(). Print the refresh_token, and copy paste in /config.ini
            Step 4 : 
        """
        self.client_id      = mk1.config.get("api_dropbox", "client_id")
        self.client_secret  = mk1.config.get("api_dropbox", "client_secret")
        self.refresh_token  = mk1.config.get("api_dropbox", "refresh_token")
        # self.auth_code      = mk1.config.get("api_dropbox", "auth_code")
        # self.refresh_token = self.get_refresh_token()

        self.access_token = self.refresh_access_token()
        self.dbx = dropbox.Dropbox(
            app_key=self.client_id,
            app_secret=self.client_secret,
            oauth2_refresh_token=self.refresh_token
        )
    
   
    

    def get_refresh_token(self):
        url = "https://api.dropboxapi.com/oauth2/token"
        data = {
            "code"              : self.auth_code,
            "grant_type"        : "authorization_code",
            "client_id"         : self.client_id,
            "client_secret"     : self.client_secret,
        }
        response = requests.post(url, data = data)

        if response.status_code == 200:
            token_data = response.json()
            return token_data.get("refresh_token")
        else:
            raise Exception(f"Failed to get refresh token: {response.text}")


    def refresh_access_token(self):
        url = "https://api.dropboxapi.com/oauth2/token"
        data = {
            "grant_type"    : "refresh_token",
            "refresh_token" : self.refresh_token,
            "client_id"     : self.client_id,
            "client_secret" : self.client_secret,
        }
        response = requests.post(url, data = data)
        if response.status_code == 200:
            token_data = response.json()
            return token_data["access_token"]
        else:
            raise Exception(f"Failed to refresh token: {response.text}")


    def list_all_folders(self):
        # Initialize list of folders
        all_folders = []
        
        # Call the Dropbox API to list the folders in the root directory
        try:
            result = self.dbx.files_list_folder('')
            all_folders.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FolderMetadata)])
            
            # If there are more folders, continue retrieving them
            while result.has_more:
                result = self.dbx.files_list_folder_continue(result.cursor)
                all_folders.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FolderMetadata)])
                
        except dropbox.exceptions.ApiError as e:
            print(f"Error retrieving folders: {e}")
        
        return all_folders

    def upload_file(self, local_path: str, dropbox_path: str):
        """Uploads a file to Dropbox."""
        with open(local_path, "rb") as f:
            self.dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))

    def download_file(self, dropbox_path: str, local_path: str):
        """Downloads a file from Dropbox."""
        metadata, res = self.dbx.files_download(dropbox_path)
        with open(local_path, "wb") as f:
            f.write(res.content)

    def list_files(self, folder_path: str = ""):
        """Lists files in a Dropbox folder."""
        result = self.dbx.files_list_folder(folder_path)
        return [entry.name for entry in result.entries]

    def delete_file(self, dropbox_path: str):
        """Deletes a file from Dropbox."""
        self.dbx.files_delete_v2(dropbox_path)


# client = DropboxClient("your_access_token")
# client.upload_file("local.txt", "/dropbox_folder/remote.txt")
# client.download_file("/dropbox_folder/remote.txt", "downloaded.txt")
# print(client.list_files("/dropbox_folder"))
# client.delete_file("/dropbox_folder/remote.txt")
