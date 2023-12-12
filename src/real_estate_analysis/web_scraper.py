"""Module for web scraping real estate data, including URL fetching, house document fetching, and data extraction."""


import ssl
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as BS
import pandas as pd
import time
import numpy as np


class HouseDocumentFetcher:
    """
    A class to fetch HTML documents of house listings.

    This class uses SSL context to bypass certain SSL certificate verification issues.
    """
    def __init__(self):
        """
        Initializes the SSL context.
        """
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE

    def fetch_house_doc(self, house_url):
        """
        Fetches the HTML document for a given house URL.

        Parameters:
        house_url (str): URL of the house listing.

        Returns:
        BeautifulSoup object: Parsed HTML document of the house listing.
        """
        req = Request(house_url, headers={"User-Agent": "Mozilla/5.0"})
        webpage = urlopen(req, context=self.ctx).read()
        house_page_doc = BS(webpage, "html.parser")
        return house_page_doc


class URLFetcher:
    """
    A class to fetch URLs from a given main URL of a house listing page.
    """
    def fetch_page_urls(main_url, house_doc_fetcher):
        """
        Fetches all related page URLs from the main URL.

        Parameters:
        main_url (str): The main URL of the house listing.
        house_doc_fetcher (HouseDocumentFetcher): An instance of HouseDocumentFetcher.

        Returns:
        list: List of all fetched URLs.
        """
        mainurldoc = house_doc_fetcher.fetch_house_doc(main_url)
        base_url = "https://www.redfin.com"
        url_list = [main_url]
        for hrefs in mainurldoc.findAll(
            "a", {"class": "clickable goToPage"}, href=True
        ):
            next_page = base_url + hrefs["href"]
            url_list.append(next_page)
        return url_list


class HouseURLExtractor:
    """
    A class to extract house listing URLs from a list of page URLs.
    """
    
    def extract_house_urls(url_list, house_doc_fetcher):
        """
        Extracts house listing URLs from the given list of page URLs.

        Parameters:
        url_list (list): List of page URLs.
        house_doc_fetcher (HouseDocumentFetcher): An instance of HouseDocumentFetcher.

        Returns:
        list: List of house listing URLs.
        """
        house_urls = []
        for url in url_list:
            doc = house_doc_fetcher.fetch_house_doc(url)
            divs = doc.find_all("div", {"class": "bottomV2"})
            for div in divs:
                href = div.a.get("href")
                if href and href.startswith("/"):
                    href = href[1:]  # Remove the leading slash
                full_url = "https://www.redfin.com/" + href
                house_urls.append(full_url)
        return house_urls


class HouseDataExtractor:
    """
    A class to extract and save property details from house listing URLs.
    """
    
    def scrape_and_save_property_details(url_list, filename, batch_size=10):
        """
        Scrapes and saves property details from the given list of URLs.

        Parameters:
        url_list (list): List of property URLs to scrape.
        filename (str): File name to save the property details.
        batch_size (int): Number of records to process before saving to file.

        Returns:
        DataFrame: DataFrame containing the scraped property details.
        """
        details_df = pd.DataFrame(columns=[...])  # Specify your columns

        for i, url in enumerate(url_list):
            try:
                house_doc = HouseDocumentFetcher().fetch_house_doc(url)
                house_dict = HouseDataExtractor.extract_house_details(house_doc)
                details_df = details_df.append(house_dict, ignore_index=True)

                if i % batch_size == 0:
                    details_df.to_csv(filename, index=False)

                time.sleep(3)  # Adjust delay as necessary

            except Exception as e:
                # Handle or log the exception
                pass

        details_df.to_csv(filename, index=False)
        return details_df

    def extract_house_details(doc):
        """
        Extracts details of a house from its HTML document.

        Parameters:
        doc (BeautifulSoup): Parsed HTML document of the house listing.

        Returns:
        dict: Dictionary containing extracted house details.
        """
        SuperCenters = [
            "Costco",
            "Target",
            "Walmart",
            "Safeway",
            "Lucky",
            "FoodMaxx",
            "Trader Joe",
            "Walgreens",
        ]
        Major_Indian_Grocery = [
            "New India",
            "Apna Bazaar",
            "India Cash And Carry",
            "Trinetra",
            "Namaste",
            "Saudagar",
            "Bharat",
            "Nilgiris",
        ]
        Major_Entertainment = ["AMC", "Cinemark", "Theater"]
        Boba = ["Bubble Tea", "Boba"]
        Healthcare = [
            "Hospital",
            "Clinic",
            "Kaiser Permanante",
            "United Health",
            "Health",
        ]

        table_dict = {}
        for i in doc.findAll(
            "div", {"class": "keyDetail font-weight-roman font-size-base"}
        ):
            x = i.findChildren("span")[0].get_text().split("\n")
            y = i.findChildren("span")[1].get_text().split()
            z = zip(x, y)
            table_dict.update(dict(list(z)))

        try:
            price = doc.find("div", attrs={"class": "statsValue"}).text.split("$")[1]
        except:
            price = np.nan
        try:
            Address = doc.find(
                "h1", attrs={"data-rf-test-id": "abp-homeinfo-homeaddress"}
            ).text
        except:
            Address = np.nan

        try:
            bed_str = doc.find("div", attrs={"data-rf-test-id": "abp-beds"}).text.split(
                "B"
            )[0]
            beds = int(bed_str)
        except:
            beds = np.nan

        try:
            str = doc.find("div", attrs={"data-rf-test-id": "abp-baths"}).text.split(
                "B"
            )[0]
            bath = float(str)
        except:
            bath = np.nan

        try:
            sqft_str = (
                doc.find("div", attrs={"data-rf-test-id": "abp-sqFt"})
                .text.split("S")[0]
                .replace(",", "")
            )
            living_sqft = float(sqft_str)
        except:
            living_sqft = np.nan

        try:
            d_score = int(
                doc.find("tspan", attrs={"class": "riskValue redOrange"}).text
            )
        except:
            d_score = np.nan

        try:
            nbd_homes = int(
                doc.find(
                    "div", attrs={"class": "MarketInsightsRegionCard--cardMetrics row"}
                )
                .text.split()[-3]
                .strip("Sale-to-List")
                .strip("#")
            )
        except:
            nbd_homes = np.nan

        try:
            wa_score = int(
                doc.findAll("div", {"class": "score inline-block not-last"})[0]
                .get_text()
                .split(" ")[0]
            )
        except:
            wa_score = np.nan
        try:
            trans_score = int(
                doc.findAll("span", {"class": "value fair"})[0].get_text()
            )
        except:
            trans_score = np.nan

        for div in doc.findAll("div", {"class": "PointOfInterestWidget"}):
            groceries = int(div.findAll("div", {"class": "Tag__text"})[1].get_text())
            services = int(div.findAll("div", {"class": "Tag__text"})[-1].get_text())
            emergency = int(div.findAll("div", {"class": "Tag__text"})[-2].get_text())
            shopping = int(div.findAll("div", {"class": "Tag__text"})[4].get_text())
            food_drink = int(div.findAll("div", {"class": "Tag__text"})[2].get_text())

        # places nearby items
        try:
            Nearby_Items = doc.find(
                "div", attrs={"class": "PointOfInterestWidget"}
            ).text
        except:
            Nearby_Items = np.nan

        # competitive score
        try:
            Competitive_Score = doc.find("div", attrs={"class": "score most"}).text
        except:
            Competitive_Score = np.nan

        # activity views
        # views, favorite, favrotie all time, x-outs, all-time x-outs, tours, all- time tours
        try:
            activity_lis = [
                activity.get_text()
                for activity in doc.findAll(
                    "span",
                    attrs={
                        "class": "count",
                        "data-rf-test-name": "activity-count-label",
                    },
                )
            ]
        except:
            activity_lis = np.nan

        try:
            view_count = activity_lis[0]
        except:
            view_count = np.nan

        try:
            fav_count_30 = activity_lis[1]
        except:
            fav_count_30 = np.nan

        try:
            fav_all_time = activity_lis[2]
        except:
            fav_all_time = np.nan

        school_dict = {}
        name_lis = [
            school.get_text()
            for school in doc.findAll(
                "div", {"class": "school-name font-size-base font-weight-bold"}
            )
        ]
        # dis_lis= [float(dis.get_text().strip("mi")) for dis in doc.findAll('div', {'class':'subsection-number'}) if "mi" in dis.get_text()]
        rating_lis = [
            school.get_text()
            for school in doc.findAll(
                "span", {"class": "rating-num font-size-base font-weight-bold"}
            )
        ]
        for x, y in zip(name_lis, rating_lis):
            school_dict[x] = y

        has_supercenter = 0
        has_major_indian_grocery = 0
        has_major_entertainment = 0
        has_indian_restaurant = 0
        has_chinese_restaurant = 0
        has_mexican_restaurant = 0
        has_boba = 0
        has_starbucks = 0
        has_healthcare_support = 0
        has_mall = 0

        try:
            if any(centre in Nearby_Items for centre in SuperCenters):
                has_supercenter = 1
        except:
            has_supercenter = 0

        try:
            if any(grocery in Nearby_Items for grocery in Major_Indian_Grocery):
                has_major_indian_grocery = 1
        except:
            has_major_indian_grocery = 0

        try:
            if any(
                entertainment in Nearby_Items for entertainment in Major_Entertainment
            ):
                has_major_entertainment = 1
        except:
            has_major_entertainment = 0

        try:
            if "Indian Restaurant" in Nearby_Items:
                has_indian_restaurant = 1
        except:
            has_indian_restaurant = 0

        try:
            if "Chinese Restaurant" in Nearby_Items:
                has_chinese_restaurant = 1
        except:
            has_chinese_restaurant = 0

        try:
            if "Mexican Restaurant" in Nearby_Items:
                has_mexican_restaurant = 1
        except:
            has_mexican_restaurant = 0

        try:
            if any(boba_tea in Nearby_Items for boba_tea in Boba):
                has_boba = 1
        except:
            has_boba = 0

        try:
            if "Starbucks" in Nearby_Items:
                has_starbucks = 1
        except:
            has_starbucks = 0

        try:
            if any(health in Nearby_Items for health in Healthcare):
                has_healthcare_support = 1
        except:
            has_healthcare_support = 0

        try:
            if "Mall" in Nearby_Items:
                has_mall = 1
        except:
            has_mall = 0

        l = {
            "List_price": price,
            "Address": Address,
            "Beds": beds,
            "Baths": bath,
            "Living_sqft": living_sqft,
            "Drought_Score": d_score,
            "Walk_score": wa_score,
            "Neighbourhood_Homes": nbd_homes,
            "Transit_score": trans_score,
            "Groceries_stores": groceries,
            "Services": services,
            "Emergency": emergency,
            "Shopping": shopping,
            "Food_and_Drink": food_drink,
            "Schools": [school_dict],
            "Competitive_Score": Competitive_Score,
            "page_view_count": view_count,
            "page_fav_count_30": fav_count_30,
            "page_fav_all_time_count": fav_all_time,
            "has_supercenter": has_supercenter,
            "has_major_indian_grocery": has_major_indian_grocery,
            "has_major_entertainment": has_major_entertainment,
            "has_indian_restaurant": has_indian_restaurant,
            "has_chinese_restaurant": has_chinese_restaurant,
            "has_mexican_restaurant": has_mexican_restaurant,
            "has_boba": has_boba,
            "has_starbucks": has_starbucks,
            "has_healthcare_support": has_healthcare_support,
            "has_mall": has_mall,
        }
        table_dict.update(l)
        return table_dict
