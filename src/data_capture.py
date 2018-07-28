from bs4 import BeautifulSoup
import json
import requests
from selenium import webdriver
import sys
from tqdm import tqdm


driver = webdriver.Chrome("data/chromedriver")
base_url = "https://www.jleague.jp"


def make_url(team):
    url = base_url + "/club/{}/day/#player".format(team)
    return url


def get_players(url):
    driver.get(url)
    html = driver.page_source.encode('utf-8')
    soup = BeautifulSoup(html, "lxml")

    player_table = soup.find("table", class_="playerDataTable")
    player_table = player_table.find("tbody")
    players = player_table.find_all("tr")

    return players


def get_player_info(team, player):
    pl_dict = {}
    info = player.find_all("td")

    pl_dict["number"] = info[0].text
    pl_dict["name"] = info[2].text
    pl_dict["position"] = info[3].text
    pl_dict["birthplace"] = info[4].text
    pl_dict["birthday"] = info[5].text
    pl_dict["height"] = info[6].text
    pl_dict["weight"] = info[7].text

    pl_dict["img"] = base_url + player.find("img")["src"]

    return pl_dict


def save_images(team, players):
    base_path = "data/JLeaguers/"
    for p in tqdm(players):
        p_dic = get_player_info(team, p)

        img = requests.get(p_dic["img"])
        p_path = "/{}_{}".format(team, p_dic["number"])
        img_path = base_path + "images" + p_path + ".jpg"
        with open(img_path, "wb") as f:
            f.write(img.content)

        json_path = base_path + "annotations" + p_path + ".json"
        with open(json_path, "wt") as f:
            json.dump(p_dic, f, indent=4, ensure_ascii=False)


def main():
    team = sys.argv[1]
    url = make_url(team)
    players = get_players(url)
    save_images(team, players)
    driver.close()


if __name__ == "__main__":
    main()
