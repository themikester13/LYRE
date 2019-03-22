# LYRE (LYRE - LYrical Real-time Encoding)

Our goal is to design a system that will allow a user to track song lyrics in real-time as the song is played.  Software will work to separate the foreground from the background signal and match lyrics with the isolated vocals in a temporally accurate manner.  The end product will be similar to the interface of a karaoke track or the feature of the Shazam app in which lyrics are highlighted as the song plays.

Our Website is located at: http://lyrictrackerkaraok.wixsite.com/LYRE

To successfully run our program:
  1. Install all python dependencies by running:
      - pip install -r requirements.txt
      - A few python dependencies might be left out. If so, install those.
  2. Install PyTorch
  3. Install ImageMagick from:
      - https://www.imagemagick.org/
  4. Get necessary permissions to use the API
  5. Use Jyputer Notebook and run Get Vocals. Requires the path to the original song
  6. Determine which is the foreground and move that .wav file to "musicfile/foreground"
  7. Navigate to the root project directory through console/terminal and run:
      - python audioToKaraoke.py "Song Title" "model name" "graph word"
      - Note: "Song Title" is the title of the .wav file located in musicfile/foreground and "model name" is either "approx" or "onset"
              - Graph word is which word to graph in the wav.
      
What we have currently:
  - API calls to google speech API
      - https://cloud.google.com/speech-to-text/
      - Status: Working
  - Word Time Stamps:
      - Status: Working
  - Syllable Time Models:
      - Approximate:
        - Status: Working
      - Onset Detection:
        - Status: Working
  - Video Generation:
      - Status: Working
      
Note: The code assumes that there is a directory heirarchy that must be followed:
  - The actual sound file needs to be put in soundfiles/background or soundfile/foreground assuming that the isolated vocals are in foreground
  - The video files will be generated and stored in movieFiles/Karaoke
Removed Commit History because accidentally commited Google Credentials
