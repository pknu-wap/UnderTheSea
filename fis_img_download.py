from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib.parse
import requests
import os 

LIMIT = 100

# Get Fish Name for Search

fish = urllib.parse.quote_plus(str(input("What are you looking for?: ")))
print(fish)

# Make url
url = "https://www.google.com/search?q=" + fish + "&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg"

print("Searching for " + fish + "images...")

# Get html text
req = requests.get(url)
html = req.text

soup = BeautifulSoup(html, 'html.parser')

# Search images in pages
images_data = soup.find_all("img")

# Get Source & Save
for i in enumerate(images_data[1:]):
    # LIMIT 
    if(i[0] >= LIMIT):
        break

    # Get Image Source
    t = urlopen(i[1].attrs['src'], None, 10)
    data = t.read()

    # Make File Name
    filename = fish + str(i[0]+1) + '.jpg'

    # Save Image
    with open(filename, "wb") as f:
        f.write(data)

    print(filename + " Img Saved")
