from icrawler.builtin import GoogleImageCrawler

def download_image(fish_name, max_num):
    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=54,
        storage={'root_dir': 'fish_image/'+fish_name})

    google_crawler.crawl(keyword=fish_name, filters=None, offset=0, max_num=max_num,
                        min_size=None, max_size=None, file_idx_offset=0)


def main():
    fish_name = input('Fish name : ')
    max_num = int(input('Number of photo : '))
    download_image(fish_name, max_num)

    print("\nDone!")

if __name__ == "__main__":
    main()
