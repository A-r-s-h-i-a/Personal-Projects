import os
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC



class igBot:
	def __init__(self, UN="from__a__distance", PW="af921909"):
		self.HOME_LINK = "https://www.instagram.com/"
		self.POPUP_SLEEP = random.uniform(16, 18)
		self.UN = UN
		self.PW = PW

	def launch_chrome(self):
		# Initialize webdriver
		self.driver = webdriver.Chrome(executable_path="C:/Users/Arshia/Desktop/Instagram Bot/ChromeDriver/chromedriver.exe")
		time.sleep(1)

	def kill_chrome(self):
		# Kill webdriver
		self.driver.quit()

	def login_ig(self):
		# Go to instagram webpage
		self.driver.get(self.HOME_LINK)
		time.sleep(random.uniform(12, 15))

		# Input username
		self.driver.find_element(By.XPATH, '//*[@id="loginForm"]/div/div[1]/div/label/input').click()
		username_field = self.driver.find_element(By.XPATH, '//*[@id="loginForm"]/div/div[1]/div/label/input')
		username_field.send_keys(self.UN)
		time.sleep(random.uniform(11, 12))

		# Input password
		self.driver.find_element(By.XPATH, '//*[@id="loginForm"]/div/div[2]/div/label/input').click()
		password_field = self.driver.find_element(By.XPATH, '//*[@id="loginForm"]/div/div[2]/div/label/input')
		password_field.send_keys(self.PW)
		time.sleep(random.uniform(11, 12))

		# Click login button
		self.driver.find_element(By.XPATH, '//*[@id="loginForm"]/div/div[3]/button/div').click()
		time.sleep(random.uniform(15, 17))

		# Jump to home
		self.driver.get(self.HOME_LINK)
		time.sleep(random.uniform(13, 15))

		# Click "Not Now" or "Cancel" on pop-ups
		try:
			self.driver.find_element(By.XPATH, "//button[contains(text(), 'Not Now')]").click()
			time.sleep(random.uniform(14, 16))
		except Exception as err:
			pass
		try:
			self.driver.find_element(By.XPATH, "//button[contains(text(), 'Cancel')]").click()
			time.sleep(random.uniform(14, 16))
		except Exception as err:
			pass

	def upload_photo(self, n, caption_text, folder_path='C:\\Users\\Arshia\\Desktop\\Instagram Bot\\Generated_Images\\'):
		# Click upload button
		time.sleep(3)
		self.driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[1]/div/div/div/div[1]/div[1]/section/nav/div[2]/div/div/div[3]/div/div[3]/div/button').click()
		time.sleep(random.uniform(17, 20))

		# Create the image path, and send it to Instagram for upload
		image_path = folder_path + str(n) + '.jpeg'
		upload_input = self.driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[2]/div/div/div[1]/div/div[3]/div/div/div/div/div/div/div/div/div[2]/div[1]/form/input')
		upload_input.send_keys(image_path)
		time.sleep(random.uniform(12, 14))

		# Click "Next", "Next"
		self.driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[2]/div/div/div[1]/div/div[3]/div/div/div/div/div/div/div/div/div[1]/div/div/div[3]/div/button').click()
		time.sleep(random.uniform(12, 14))
		self.driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[2]/div/div/div[1]/div/div[3]/div/div/div/div/div/div/div/div/div[1]/div/div/div[3]/div/button').click()
		time.sleep(random.uniform(12, 14))

		# Enter caption text
		caption_input = self.driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[2]/div/div/div[1]/div/div[3]/div/div/div/div/div/div/div/div/div[2]/div[2]/div/div/div/div[2]/div[1]/textarea')
		caption_input.send_keys(caption_text)
		time.sleep(random.uniform(13, 16))

		# Click "Submit" to submit image
		self.driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[2]/div/div/div[1]/div/div[3]/div/div/div/div/div/div/div/div/div[1]/div/div/div[3]/div/button').click()
		time.sleep(random.uniform(15, 19))

		# Return to home page
		self.driver.get(self.HOME_LINK)
		time.sleep(random.uniform(11, 13))

	def return_ig_home(self):
		# Return to Instagram home page
		self.driver.get(self.HOME_LINK)
		time.sleep(random.uniform(11, 13))


if __name__ == "__main__":
	# Launch Chrome and login to Instagram
	bot = igBot()
	bot.launch_chrome()
	print("\nChrome Launched\n")
	bot.login_ig()
	print("\nLogged In to Instagram\n")

	# Upload an Image
	bot.upload_photo(n=0, caption_text="Test Text")
	print("\nUploaded an Image!\n")

	# Before ending the program, kill the app-controlled Chrome
	bot.kill_chrome()
	print("\nKilled Chrome\n")