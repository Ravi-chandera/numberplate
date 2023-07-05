import requests

# Specify the path to the video file
video_file_path = "out.mp4"

# Upload the video file to file.io
with open(video_file_path, "rb") as f:
    response = requests.post("https://file.io/?expires=1w", files={"file": f})

# Retrieve the shareable link from the response
response_data = response.json()
video_download_link = response_data["link"]

# Display the download link in Streamlit
st.markdown(f"**[Download the Video]({video_download_link})**")
