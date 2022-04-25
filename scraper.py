from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
import pandas as pd
import argparse
import os


# Function to parse command line arguments at script start
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Script for scrapping information's from rough guide website and saving them to csv.")

    parser._action_groups.pop()
    requirements = parser.add_argument_group('requirements')
    requirements.add_argument('-s',
                         dest="source_path",
                         type=str,
                         help="Path to the csv file containing the links to be scraped.",
                         required=True)

    args = parser.parse_args()

    source_path = args.source_path
    if not source_path.endswith(".csv"):
        print('\n\nInvalid use. You need to pass the path of the csv file. For more information see help '
              'accessible via flag "-h"\n\n')
        os.system('python scraper.py -h')
        quit()
    return args


# Function for read in csv with web-links to scrape
def read_links(source):
    datatable = pd.read_csv(source)
    raw_links = datatable['roughguide link'].tolist()
    return raw_links


# Function for skipping cookie banner (not needed)
def skip_cookie():
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "cf2Lf6")))
    driver.find_element(By.CLASS_NAME, 'cf2Lf6').click()
    print('>> Cookie-Banner successful skipped!')


# Function for extracting place name of webpage
def get_place():
    raw_place = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[4]/div/section[1]/div/div/main/div/section/h1').text
    #print(raw_place)
    return raw_place


# Function for extracting place description of webpage
def get_content():
    content_text = ''
    content_element = driver.find_element(By.CLASS_NAME, 'DestinationPageContent')
    raw_text = content_element.find_elements(By.TAG_NAME, 'p')
    for paragraph in raw_text:
        content_text += str(' ' + paragraph.text)
    content_text = ' '.join(content_text.splitlines())
    content_text = content_text.replace(';', ' -')
    content_text = content_text.replace("In-depth, easy-to-use travel guides filled with expert advice.", '')
    content_text = content_text.replace("Use Rough Guides' trusted partners for great rates", '')
    content_text = content_text.strip()
    #print(content_text)
    return content_text


# Function for saving scraped data to csv
def save_to_csv(lst, dest_path):
    data = pd.DataFrame(lst, columns=['Link', 'Place', 'Content'])
    destination_file_path = dest_path + "/results.csv"
    data.to_csv(destination_file_path, header=['Link', 'Place', 'Content'], encoding='utf-8-sig')


if __name__ == '__main__':
    args = parse_args()
    source_path = args.source_path
    dest_path = source_path.rsplit('/', 1)[0]
    count = 1
    lst = []
    start_time = time.time()

    try:
        # Read in csv with page links
        print('Reading links from csv..')
        links = read_links(source_path)

        # Start driver
        print('Setting up chrome webdriver..')
        driver = webdriver.Chrome()
        WebDriverWait(driver, 10)

        # Iterate over links
        print('Starting scraping data from pages..')
        for link in links:
            # Open page
            print('Processing scraping of page {0} of {1}'.format(count, len(links)))
            driver.get(link)

            # Get header text (place name) of page
            place = get_place()

            # Get content text (place description) of page
            content = get_content()

            # Append place and content to list
            lst.append([link, place, content])

            # Wait 1 sec and repeat procedure for next page
            time.sleep(1)
            count += 1

        # Save list as dataframe and export to csv
        print('Exporting scraped data to csv')
        save_to_csv(lst, dest_path)

        # Exit driver
        print('Finished scraping! Took %s min to scrape' % round((time.time() - start_time) / 60), 2)
        time.sleep(2)
        driver.quit()


    except KeyboardInterrupt:
        # Catch Keyboard Interrupt
        print('\nThe programm was interrupted by keyboard')
        exit()

    except Exception as e:
        # Catch Error and safely close script
        print('\nThe program has encountered an error')
        exit()