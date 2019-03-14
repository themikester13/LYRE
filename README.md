# LYRE (LYRE - LYrical Real-time Encoding)

Our goal is to design a system that will allow a user to track song lyrics in real-time as the song is played.  Software will work to separate the foreground from the background signal and match lyrics with the isolated vocals in a temporally accurate manner.  The end product will be similar to the interface of a karaoke track or the feature of the Shazam app in which lyrics are highlighted as the song plays.

To successfully run our program:
  1. Install all python dependencies by running:
      - pip install -r requirements.txt
      - A few python dependencies might be left out. If so, please lmk what they are
  2. Install ImageMagick from:
      - https://www.imagemagick.org/
  4. Get necessary permissions to use the API
      - Just a json file
  3. Navigate to the root project directory through console/terminal and run:
      - python text2Speech.py OR python3 text2Speech.py
      
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
