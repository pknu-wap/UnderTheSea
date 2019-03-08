import urllib.request
from urllib.request import Request, urlopen
from urllib.request import URLError, HTTPError
from urllib.parse import quote
import http.client
from http.client import IncompleteRead
http.client._MAXHEADERS = 1000
import os
import argparse
import ssl
import json

class fish_images_download:
    def __init__(self):
        pass

    # Make Directories
    def create_directories(self,main_directory, dir_name):
        try:
            if not os.path.exists(main_directory):
                os.makedirs(main_directory)
                path = str(dir_name)
                sub_directory = os.path.join(main_directory, path)
                if not os.path.exists(sub_directory):
                    os.makedirs(sub_directory)
            else:
                path = str(dir_name)
                sub_directory = os.path.join(main_directory, path)
                if not os.path.exists(sub_directory):
                    os.makedirs(sub_directory)
                
        except OSError as e:
            if e.errno != 17:
                raise
            pass
        return

    # Make Search URL
    def build_search_url(self,search_term):
        url = 'https://www.google.com/search?q=' + quote(
            search_term) + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'

        return url

    # Get Next Tab
    def get_next_tab(self,s):
        start_line = s.find('class="dtviD"')
        if start_line == -1:
            end_quote = 0
            link = "no_tabs"
            return link,'',end_quote
        else:
            start_line = s.find('class="dtviD"')
            start_content = s.find('href="', start_line + 1)
            end_content = s.find('">', start_content + 1)
            url_item = "https://www.google.com" + str(s[start_content+6:end_content])
            url_item = url_item.replace('&amp;', '&')

            start_line_2 = s.find('class="dtviD"')
            start_content_2 = s.find(':', start_line_2 + 1)
            end_content_2 = s.find('"', start_content_2 + 1)
            url_item_name = str(s[start_content_2 + 1:end_content_2])

            return url_item,url_item_name,end_content

    # Get Next Images
    def get_next_images(self,s):
        start_line = s.find('rg_meta notranslate')
        if start_line == -1:  # If no links are found then give an error!
            end_quote = 0
            link = "no_links"

            return link, end_quote
        else:
            start_line = s.find('class="rg_meta notranslate">')
            start_object = s.find('{', start_line + 1)
            end_object = s.find('</div>', start_object + 1)
            object_raw = str(s[start_object:end_object])
  
            try:
                object_decode = bytes(object_raw, "utf-8").decode("unicode_escape")
                final_object = json.loads(object_decode)
            except:
                final_object = ""

            return final_object, end_object

    # Get All Images
    def get_all_images(self,page,main_directory,dir_name,limit):
        images = []
        abs_path = []
        errorCount = 0
        i = 0
        count = 1
        while count < limit+1:
            object, end_content = self.get_next_images(page)
            if object == "no_links":
                break
            elif object == "":
                page = page[end_content:]
            else:
                #format the item for readability
                object = self.format_object(object)

                #download the images
                download_status,download_message,return_image_name,absolute_path = self.download_image(object['image_link'],object['image_format'],main_directory,dir_name,count)
                print(download_message)
                if download_status == "success":
                    count += 1
                    object['image_filename'] = return_image_name
                    images.append(object)  # Append all the links in the list named 'Links'
                    abs_path.append(absolute_path)
                else:
                    errorCount += 1

                page = page[end_content:]
            i += 1
        if count < limit:
            print("\nDonwloaded Images: " + str(count-1))
        return images,errorCount,abs_path

    # Format  
    def format_object(self,object):
        formatted_object = {}
        formatted_object['image_format'] = object['ity']
        formatted_object['image_height'] = object['oh']
        formatted_object['image_width'] = object['ow']
        formatted_object['image_link'] = object['ou']
        formatted_object['image_description'] = object['pt']
        formatted_object['image_host'] = object['rh']
        formatted_object['image_source'] = object['ru']
        formatted_object['image_thumbnail_url'] = object['tu']
        return formatted_object

    # Download Images
    def download_image(self,image_url,image_format,main_directory,dir_name,count):
        try:
            req = Request(image_url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
            try:
                timeout = 10

                response = urlopen(req, None, timeout)
                data = response.read()
                response.close()

                image_name = dir_name.replace(" ","_") + "_" + str(count) + '.jpg'
                image_name = image_name.lower()

                path = main_directory + "/" + dir_name + "/" + image_name

                try:
                    output_file = open(path, 'wb')
                    output_file.write(data)
                    output_file.close()
                    absolute_path = os.path.abspath(path)
                except OSError as e:
                    download_status = 'fail'
                    download_message = "OSError on an image...trying next one..." + " Error: " + str(e)
                    return_image_name = ''
                    absolute_path = ''

                #return image name back to calling method to use it for thumbnail downloads
                download_status = 'success'
                download_message = image_name
                return_image_name = image_name

            except UnicodeEncodeError as e:
                download_status = 'fail'
                download_message = "UnicodeEncodeError on an image...trying next one..." + " Error: " + str(e)
                return_image_name = ''
                absolute_path = ''

            except URLError as e:
                download_status = 'fail'
                download_message = "URLError on an image...trying next one..." + " Error: " + str(e)
                return_image_name = ''
                absolute_path = ''

        except HTTPError as e:  # If there is any HTTPError
            download_status = 'fail'
            download_message = "HTTPError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        except URLError as e:
            download_status = 'fail'
            download_message = "URLError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        except ssl.CertificateError as e:
            download_status = 'fail'
            download_message = "CertificateError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        except IOError as e:  # If there is any IOError
            download_status = 'fail'
            download_message = "IOError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        except IncompleteRead as e:
            download_status = 'fail'
            download_message = "IncompleteReadError on an image...trying next one..." + " Error: " + str(e)
            return_image_name = ''
            absolute_path = ''

        return download_status,download_message,return_image_name,absolute_path

    # Download Image
    def download(self,keyword):
        
        search_keyword = keyword
        main_directory = "fish_images"
        limit = 1000
                    
        print("\nSearching for " + search_keyword + " images...\n")

        # Get url
        dir_name = search_keyword
        self.create_directories(main_directory,dir_name)
        url = self.build_search_url(search_keyword)
        
        # Get html
        headers = {}
        headers['User-Agent'] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.1 Safari/605.1.15"
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req)
        html = str(resp.read())
        
        print("Starting Download...")

        # Get All Images
        images,errorCount,abs_path = self.get_all_images(html,main_directory,dir_name,limit)

        print("\nErrors: " + str(errorCount) + "\n")

        return abs_path

def main():
    fish = urllib.parse.quote_plus(str(input("Enter Name: ")))

    response = fish_images_download()
    paths = response.download(fish)

    print("\nDone!")

if __name__ == "__main__":
    main()
