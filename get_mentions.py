# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:51:27 2022

@author: R.P.L. Azevedo
"""

import asyncio
import datetime as dt
import re

import aiohttp
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from bs4 import BeautifulSoup as bs


def number_arts_with_name(name: str, list_of_art: list[str]) -> int:
    """Checks in how many articles a person is mentioned"""
    name_split = name.split()
    search_pattern = r'(' + name_split[0] + \
                     r') *(?:\w*\.?-? *){0,4} *(' + name_split[-1] + r')'
    name_search = re.compile(search_pattern)
    n = 0
    for art in list_of_art:
        result = name_search.findall(art)
        if result:
            n += 1
    return n


async def get_article(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as resp:
        body = await resp.text()
    article_clean = await clean_article(body, session)
    return article_clean


async def clean_article(html: str, session: aiohttp.ClientSession) -> str:
    soup = bs(html, "lxml")
    article_body = soup.find('div', {"class": "entry-content"})
    url_el = soup.find('a', string="Saber mais »")
    if url_el:
        url = url_el['href']
        async with session.get(url) as resp:
            resp = await resp.text()
            soup = bs(resp, "lxml")
            article_body = soup.find('div', {"class": "newsContent"})
    article_text = article_body.text
    article_clean = article_text.partition(
        "\nGrupo de Comunicação de Ciência")[0]
    article_clean = article_clean.partition("\nContactos")[0]
    article_clean = article_clean.partition("\n Contactos")[0]
    return article_clean


async def get_article_urls(session: aiohttp.ClientSession, url: str) \
        -> list[tuple[str, str]]:
    async with session.get(url) as resp:
        page = await resp.text()
        soup = bs(page, "lxml")
        articles = soup.find_all('article', {"class": "genaral-post-item"})
        date_url_list = []
        for article in articles:
            url = article.find('a')['href']
            date_element = article.find('time', {"class": "entry-date"})
            if date_element is not None:
                date = date_element.get('datetime')
            else:
                date_element = article.find(
                    'div', {"class": "event-entry-datetime"})
                if date_element is not None:
                    re_date = re.findall(r'\d{4}', date_element.text)
                    if re_date:
                        date = re_date[0]
                    else:
                        date = str(dt.datetime.now().year)
                else:
                    date = ''
            date_url_list.append((date, url))
        return date_url_list


async def get_urls(main_url: str) -> list[tuple[str, str]]:
    async with (aiohttp.ClientSession(headers={"Accept-Language": "pt-Pt,pt"})
                as session):
        tasks = []
        text = requests.get(main_url).text
        soup = bs(text, "lxml")
        n_pages = int(soup.find_all('a', {"class": "page-numbers"})[-2].text)
        if main_url[-1] != '/':
            main_url = main_url + '/'
        for number in range(1, n_pages + 1):
            url = main_url + f'page/{number}'
            tasks.append(asyncio.create_task(get_article_urls(session, url)))
        temp_date_url_list = await asyncio.gather(*tasks)
        date_url_list = []
        for item in temp_date_url_list:
            date_url_list += item[:]
        return date_url_list


async def get_articles_aslist(data: pd.DataFrame):
    async with (aiohttp.ClientSession(headers={"Accept-Language": "pt-Pt,pt"})
                as session):
        tasks = []
        for url in data["url"]:
            tasks.append(asyncio.create_task(get_article(session, url)))
        temp = await asyncio.gather(*tasks)
        articles = [article for article in temp]
        return articles


async def get_data(main_url: str, table_loc: str, file_prefix: str):
    news_df = pd.DataFrame(await
                           get_urls(main_url),
                           columns=('date', 'url'))
    news_df["date"] = pd.to_datetime(news_df["date"])
    article_list = await get_articles_aslist(news_df)
    news_df["text"] = article_list

    names_df = pd.read_csv(table_loc)
    current_year = dt.datetime.now().year
    for year in range(2014, current_year + 1):
        temp_list = []
        for name in names_df["Name"]:
            n_mentions = number_arts_with_name(
                name, news_df["text"][news_df["date"].dt.year == year])
            temp_list.append(n_mentions)
        names_df[str(year)] = temp_list

    cols = [str(year) for year in range(2014, current_year + 1)]
    names_df['total'] = names_df[cols].sum(axis=1)
    names_df['gender'].fillna('unknown', inplace=True)
    names_df.to_csv(f"{file_prefix}_mention_data.csv")
    return names_df


def process_data(data: pd.DataFrame, file_prefix: str):
    f, ax = plt.subplots(1, 1, figsize=(15, 20))
    sns.barplot(data=data[data['total'] != 0],
                y="Name", x="total", hue='gender', ax=ax)
    plt.title(f"Mentions in {file_prefix}")
    f.tight_layout()
    f.savefig(f"{file_prefix}_total_mentions")

    year_list = [str(year) for year in range(2014, dt.datetime.now().year + 1)]
    m_list = [data[data["gender"] == "m"][year].sum()
              for year in year_list]
    f_list = [data[data["gender"] == "f"][year].sum()
              for year in year_list]
    # m_total = data[data["gender"] == "m"]["total"].sum()
    # f_total = data[data["gender"] == "f"]["total"].sum()
    d = {'year': year_list, 'm': m_list, 'f': f_list}

    gender_mentions_df = pd.DataFrame(data=d)
    gender_mentions_df.head()
    gender_mentions_df['m (%)'] = 100 * gender_mentions_df['m'] / (
            gender_mentions_df['m'] + gender_mentions_df['f'])
    gender_mentions_df['f (%)'] = 100 * gender_mentions_df['f'] / (
            gender_mentions_df['m'] + gender_mentions_df['f'])
    gender_mentions_df.to_csv(f"{file_prefix}_gender_mention_data.csv")

    gender_mentions_df = gender_mentions_df[(gender_mentions_df['m'] > 0) |
                                            (gender_mentions_df['f'] > 0)]
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex="True")
    gender_mentions_df.plot(x='year', y=['m (%)', 'f (%)'], ax=ax1)
    ax1.set_ylabel('% of mentions')
    gender_mentions_df.plot(x='year', y=['m', 'f'], ax=ax2)
    ax2.set_ylabel('No. of mentions')
    f.suptitle(f"Mentions in {file_prefix}")
    f.tight_layout()
    f.savefig(f'{file_prefix}_mentions')


def main():
    ia_table = "IATeam.csv"
    main_url_list = [r"https://divulgacao.iastro.pt/pt/noticias",
                     r"https://divulgacao.iastro.pt/pt/o-universo",
                     r"https://divulgacao.iastro.pt/pt/atividades-publicas"]
    prefix_list = ["noticias", "o_universo", "eventos"]

    for main_url, prefix in zip(main_url_list, prefix_list):
        names_df = asyncio.run(get_data(main_url, ia_table, prefix))
        process_data(names_df, prefix)


def profile():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats(filename="profile.prof")


if __name__ == "__main__":
    main()
