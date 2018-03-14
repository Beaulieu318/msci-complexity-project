import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
import urllib.request
import ssl

from selenium import webdriver
from selenium.webdriver.common import action_chains, keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from msci.utils import log_bin


class WestfieldURLS:

	def __init__(self, url='https://uk.westfield.com/london/stores/all-stores'):
		self.urls = {}
		self.browser = webdriver.Chrome()
		self.browser.get(url)
		self.shop_elements = self.browser.find_elements_by_xpath(
			"//*[@class='store-tile__primary js-tile-primary']"
			)
		self.site_links = {
			e.get_attribute('data-track-label'): e.get_attribute('href') for e in self.shop_elements
			}
		

	def back(self):
		self.browser.back()


	def search_shop_website(self, shop_name):
		if '&' in shop_name:
			shop_name = shop_name.replace('&', '%26')
		self.browser.find_element_by_tag_name('body').send_keys(keys.Keys.COMMAND + 't') 
		self.browser.get('https://www.google.co.uk/search?q=' + shop_name + '+shop')
		wait = WebDriverWait(self.browser, 50).until(
			EC.visibility_of_element_located((By.CLASS_NAME, "r"))
			)
		g_result = self.browser.find_elements_by_class_name("r")
		attempt = True
		it = 0
		while attempt:
			if 'href' in g_result[it].get_attribute("innerHTML"):
				ht = g_result[it].get_attribute("innerHTML")
				attempt = False
				link = ht[ht.find("href")+6:ht.find("ping")-2]
				self.browser.find_element_by_tag_name('body').send_keys(keys.Keys.COMMAND + 'w')
				return link
			elif it == len(g_result):
				return 'no_link'
			else:
				it += 1


	def get_shop_website(self, link):
		self.browser.find_element_by_tag_name('body').send_keys(keys.Keys.COMMAND + 't') 
		self.browser.get(link)
		#Find link to shop website from directory entry on Westfield's website
		try:
			url_element = self.browser.find_element_by_xpath(
				"//a[@target='_blank' and @rel='nofollow']"
				)
			url = url_element.get_attribute('href')
			self.browser.find_element_by_tag_name('body').send_keys(keys.Keys.COMMAND + 'w') 
			return url 
		except:
			return 'no_link'
			pass
		

	def get_all_shop_urls(self, search=True):
		for e in self.site_links:
			print(e)
			if search:
				url = self.search_shop_website(e)
			else:
				url = self.get_shop_website(self.site_links[e])
			self.urls[e] = url
		return self.urls



class Scraper:

	def __init__(self, url='https://tagcrowd.com/'):
		self.url = url
		self.browser = webdriver.Chrome()
		self.browser.get(self.url)
		self.first_entry = True
		

	def enter_url(self, shop_url):
		if self.first_entry:
			web_section = self.browser.find_element_by_id("src_control1")
			web_section.click()
		try:
			element = self.browser.find_element_by_name("url_file")
			wait = WebDriverWait(self.browser, 5).until(
				EC.visibility_of(element)
			)
			# print('visible')
		finally:
			element.clear()
			element.send_keys(shop_url)
			element.submit()
			try:
				self.browser.find_element_by_id("error_text")
				return 'link_void'
			except:
				load_wait = WebDriverWait(self.browser, 10).until(
					EC.visibility_of_element_located((By.ID, "htmltagcloud"))
				)


	def extract_keywords(self):
		keyword_dictionary = {}
		complete = False
		word_id = 0
		while complete == False:
			try:
				self.browser.find_element_by_id(str(word_id))
				word = self.browser.find_element_by_id(str(word_id))
				word_string = word.get_attribute("innerHTML")
				weight_string = word.get_attribute("class")
				kword = word_string[word_string.find(">") + 1: word_string.find("</a")]
				kweight = int(weight_string[weight_string.find("ud") + 2:])
				keyword_dictionary[kword] = kweight
				word_id += 1 
			except:
				complete = True
				pass
		return keyword_dictionary


def westfield_url_dictionary():
	west = WestfieldURLS()
	urls = west.get_all_shop_urls()
	west.browser.close()
	return urls


keywords = {}	


def westfield_keyword_dictionary(url_dictionary, store_list):
	"""
	Function to extract all keywords from url_dictionary using TagCloud.
	NOTE: TagCloud can not read some websites and automatically 
	redirects to a maintenance page when this is the case. 
	This has not yet been accounted for in the code.
	"""
	TC = Scraper()
	for url in url_dictionary:
		if url in store_list:
			print(url_dictionary[url])
			if url_dictionary[url] != 'no_link':
				TC.enter_url(url_dictionary[url])
				kw = TC.extract_keywords()
				TC.first_entry = False
				keywords[url] = kw
			else:
				keywords[url] = 'no_link'
	TC.browser.close()
	return keywords


def keyword_distribution(keyword_dictionary, words_to_remove):
	#get rid of empty dictionaries for which keywords could not be found
	kwd = {i: keyword_dictionary[i] for i in keyword_dictionary if len(keyword_dictionary[i]) != 0}

	stores = list(kwd.keys())
	all_keywords = list(set(np.sum([list(kwd[i].keys()) for i in kwd])))

	#allows removal of words such as 'wwww' or generic terms such as 'policy', 'about' etc 
	desired_keywords = [i for i in all_keywords if i not in words_to_remove]

	#nodes of bipartite graph
	store_nodes = [(i, dict(node_type = 'store', label = i)) for i in stores]
	keyword_nodes = [(i, dict(node_type = 'keyword', label = i)) for i in desired_keywords]

	#construct directional graph
	KN = nx.DiGraph()

	KN.add_nodes_from(store_nodes)
	KN.add_nodes_from(keyword_nodes)

	keyword_edges = []

	for store in kwd:
		for word in kwd[store]:
			if word in desired_keywords:
				keyword_edges.append((store, word, kwd[store][word]))

	KN.add_weighted_edges_from(keyword_edges)

	return kwd, desired_keywords, KN


def graph_to_GraphML(G, name='west_shop_network.graphml'):
	nx.write_graphml(G, name)


def bin(data, bin_start=1., first_bin_width=1.4, a=1.6, drop_zeros=True):
	return log_bin.log_bin(data, bin_start, first_bin_width, a, drop_zeros=drop_zeros)


def word_network_degree_distribution(G, store_list, keyword_list, plot=True, degree_type='keyword'):
	degrees = list(nx.degree(G))
	keyword_degrees = [i for i in degrees if i[0] in keyword_list]
	store_degrees = [i for i in degrees if i[0] in store_list]
	degrees = {'keyword': keyword_degrees, 'store': store_degrees}
	if plot:
		degree_values = [i[1] for i in degrees[degree_type]]
		hist, no_bins = np.histogram(degree_values, bins=50, normed=True)
		center = (no_bins[:-1] + no_bins[1:]) / 2
		lb_centers, lb_counts = bin(degree_values)
		fig = plt.figure()
		plt.scatter(np.log10(center), np.log10(hist), color='b')
		plt.plot(np.log10(lb_centers), np.log10(lb_counts), color='r', linestyle='dashed')
		plt.xlabel('Keyword In-Degree')
		plt.ylabel('Probability')
		fig.show()
		print(lb_centers, lb_counts)
	return keyword_degrees, store_degrees

