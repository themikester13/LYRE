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
  3. Navigate to the root project directory and run:
      - python text2Speech.py OR python3 text2Speech.py
