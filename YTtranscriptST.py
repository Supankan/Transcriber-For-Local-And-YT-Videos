import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        organized_transcript = ""
        buffer_text = ""
        last_timestamp = 0.0

        for entry in transcript:
            timestamp = entry['start']
            text = entry['text'].strip()

            buffer_text += " " + text

            sentences = sent_tokenize(buffer_text.strip())
            if sentences:
                organized_transcript += f"[{last_timestamp:.2f}s]: {sentences[0]}\n\n"
                buffer_text = " ".join(sentences[1:])

            last_timestamp = timestamp

        if buffer_text.strip():
            organized_transcript += f"[{last_timestamp:.2f}s]: {buffer_text.strip()}\n\n"

        return organized_transcript.strip()
    except Exception as e:
        return str(e)


def main():
    st.title("YouTube Transcript Fetcher")

    # Initialize session state
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""

    # Input for YouTube URL
    url = st.text_input("Enter the YouTube video URL:", "")

    # Button to fetch transcript
    if st.button("Fetch Transcript"):
        if url:
            video_id = url.split("v=")[-1]
            st.write("Fetching transcript...")
            st.session_state.transcribed_text = get_youtube_transcript(video_id)

            if "Error" in st.session_state.transcribed_text:
                st.error(f"Failed to fetch the transcript: {st.session_state.transcribed_text}")
            else:
                st.success("Transcript fetched.")

    # Display transcript and download button
    if st.session_state.transcribed_text:
        st.text_area("Transcribed Text", st.session_state.transcribed_text, height=300, max_chars=None)

        st.download_button(
            label="Download Transcript",
            data=st.session_state.transcribed_text,
            file_name="transcribed_text.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please enter a valid YouTube URL and fetch the transcript.")


if __name__ == "__main__":
    main()
