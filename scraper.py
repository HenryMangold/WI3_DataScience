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
    requirements.add_argument('-d',
                              dest="dest_path",
                              type=str,
                              help="Path to directory where the scraped data as csv file should be saved.",
                              required=True)
    args = parser.parse_args()

    source_path = args.source_path
    if not source_path.endswith(".csv"):
        print('\n\nInvalid use of -s flag. You need to pass the path of the csv file. For more information see help '
              'accessible via flag "-h"\n\n')
        os.system('python scraper.py -h')
        quit()

    dest_path = args.dest_path
    if dest_path.endswith(".csv"):
        print('\n\nInvalid use of -d flag. You need to pass the path of the directory where the csv file (containing '
              'the scraped data) should be saved. For more information see help accessible via flag "-h"\n\n')
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
    # print(raw_place)
    return raw_place


def get_place_h2(title_id):
    raw_place = driver.find_element(By.ID, title_id).text
    return raw_place


# Function for extracting place description of webpage
def get_full_content():
    # list of substrings that appear in elements that should not be saved but still are in the only container that
    # that contains everything else. They can't be excluded in clean_content_text function because part of the text
    # changes for each location.
    substring_list = ['Great places to visit in',
                      'Related tailor-made travel itineraries for',
                      'Discover more places related to',
                      'The Rough Guide to',
                      'Find even more inspiration for']
    content_text = ''
    content_element = driver.find_element(By.CLASS_NAME, 'DestinationPageContent')
    raw_text = content_element.find_elements(By.TAG_NAME, 'p, h2, h3, h4')
    for element in raw_text:
        if not any(substring in element.text for substring in substring_list):
            content_text += str(' ' + element.text.strip())
    # print(content_text)
    return content_text


def get_content_between_headlines(title_id):
    # h2 element with id that contains title (all lower case and - instead of ' ')
    content_text = ''
    xpath = '//*[@id="{0}"]'.format(title_id)
    start_element = driver.find_element(By.XPATH, xpath)
    next_elements = start_element.find_elements(By.XPATH, xpath + '/following-sibling::*')

    for element in next_elements:
        if element.tag_name == 'h2':
            # break while loop if next element is a h2
            break
        content_text += str(' ' + element.text.strip())
    return content_text


def clean_content_text(content_text):
    content_text = ' '.join(content_text.splitlines())
    content_text = content_text.replace(';', ' -')
    content_text = content_text.replace("Book your individual trip, stress-free with local travel experts", '')
    content_text = content_text.replace("In-depth, easy-to-use travel guides filled with expert advice.", '')
    content_text = content_text.replace("Use Rough Guides' trusted partners for great rates", '')
    content_text = content_text.replace("Continue reading to find out more about...", '')
    content_text = content_text.replace("Planning on your own? Prepare for your trip", '')
    content_text = content_text.strip()
    return content_text


# Function for saving scraped data to csv
def save_to_csv(lst, destination_file_path):
    data = pd.DataFrame(lst, columns=['Link', 'Place', 'Content'])
    data.to_csv(destination_file_path, header=['Link', 'Place', 'Content'], encoding='utf-8-sig')


if __name__ == '__main__':
    args = parse_args()
    source_path = args.source_path
    raw_dest_path = args.dest_path
    if raw_dest_path.endswith('"'):
        raw_dest_path = raw_dest_path[:-1]
    dest_path = raw_dest_path + "\\" + "results.csv"
    # print("Destination Path: ", dest_path)
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

            if '#' in link:
                # This link only refers to a part of the page.
                # Only content between the h2 Headline and the next h2 should be extracted
                title_id = link.split('#')[1]

                # get place name
                place = get_place_h2(title_id)

                # get content until next h2
                content_raw = get_content_between_headlines(title_id)
                content = clean_content_text(content_raw)

            else:
                # Get header text (place name) of page
                place = get_place()

                # Get content text (place description) of page
                content_raw = get_full_content()
                content = clean_content_text(content_raw)

            # Append place and content to list
            lst.append([link, place, content])

            # Wait 1 sec and repeat procedure for next page
            time.sleep(1)
            count += 1

        # Save list as dataframe and export to csv
        print('Exporting scraped data to csv. Path of csv is following: ', dest_path)
        save_to_csv(lst, dest_path)

        # Exit driver
        print('Finished scraping! Took %s min to scrape' % round(((time.time() - start_time) / 60), 2))
        time.sleep(2)
        driver.quit()


    except KeyboardInterrupt:
        # Catch Keyboard Interrupt
        print('\nThe programm was interrupted by keyboard')
        exit()

    except Exception as e:
        print(e)
        # Catch Error and safely close script
        print('\nThe program has encountered an error')
        print('trying to Export scraped data to csv')
        save_to_csv(lst, dest_path)
        print('done')
        exit()