import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import chromedriver_py

class DBADRSeleniumDownloader:
    def __init__(self, download_dir=None):
        """
        Initialize the downloader with chromedriver-py
        """
        self.download_dir = download_dir or str(Path.home() / "Downloads")
        self.driver = None
        
        # Ensure download directory exists
        os.makedirs(self.download_dir, exist_ok=True)
        
    def setup_driver(self):
        """
        Setup Chrome driver using chromedriver-py
        """
        chrome_options = Options()
        
        # Set download preferences
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "safebrowsing.disable_download_protection": True,
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Additional Chrome options for better compatibility
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Uncomment the next line if you want to run headless (without browser window)
        # chrome_options.add_argument("--headless")
        
        # Use chromedriver-py to get the driver path
        chrome_driver_path = chromedriver_py.binary_path
        print(f"Using ChromeDriver from: {chrome_driver_path}")
        
        # Create service with the chromedriver path
        service = Service(chrome_driver_path)
        
        # Initialize the driver
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        return self.driver
    
    def wait_for_download_complete(self, timeout=30):
        """
        Wait for download to complete by checking the downloads folder
        """
        print("Waiting for download to complete...")
        
        # Get initial files in download directory
        initial_files = set(os.listdir(self.download_dir))
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_files = set(os.listdir(self.download_dir))
            new_files = current_files - initial_files
            
            # Check if any new files don't end with .crdownload (Chrome's temp download extension)
            completed_files = [f for f in new_files if not f.endswith('.crdownload')]
            
            if completed_files:
                for file in completed_files:
                    if file.endswith(('.xlsx', '.xls')):
                        print(f"Download completed: {file}")
                        return os.path.join(self.download_dir, file)
            
            time.sleep(1)
        
        print("Download timeout reached")
        return None
    
    def download_excel_export(self, url, wait_time=15):
        """
        Download Excel export from the Deutsche Bank ADR page using Selenium
        
        Args:
            url (str): The DB ADR page URL
            wait_time (int): Maximum time to wait for elements
        """
        try:
            # Setup driver
            self.setup_driver()
            
            # Navigate to the page
            print(f"Navigating to: {url}")
            self.driver.get(url)
            
            # Wait for page to load completely
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Give page extra time to load JavaScript
            time.sleep(3)
            
            print("Page loaded. Looking for Excel export button...")
            
            # Multiple strategies to find the Excel export button
            excel_selectors = [
                # Text-based searches
                "//a[contains(text(), 'Excel')]",
                "//button[contains(text(), 'Excel')]",
                "//a[contains(text(), 'Export')]",
                "//button[contains(text(), 'Export')]",
                "//a[contains(text(), 'Download')]",
                "//button[contains(text(), 'Download')]",
                
                # Attribute-based searches
                "//a[contains(@href, '.xlsx')]",
                "//a[contains(@href, '.xls')]",
                "//a[contains(@href, 'excel')]",
                "//a[contains(@href, 'export')]",
                
                # Class-based searches
                "//button[contains(@class, 'excel')]",
                "//a[contains(@class, 'excel')]",
                "//button[contains(@class, 'export')]",
                "//a[contains(@class, 'export')]",
                "//button[contains(@class, 'download')]",
                "//a[contains(@class, 'download')]",
                
                # Title and data attribute searches
                "//button[contains(@title, 'Excel')]",
                "//a[contains(@title, 'Excel')]",
                "//button[contains(@title, 'Export')]",
                "//a[contains(@title, 'Export')]",
                "//*[contains(@data-export, 'excel')]",
                "//*[contains(@onclick, 'excel')]",
                "//*[contains(@onclick, 'export')]",
            ]
            
            excel_element = None
            found_selector = None
            
            # Try to find Excel export element
            for selector in excel_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements:
                        # Filter out hidden elements
                        visible_elements = [elem for elem in elements if elem.is_displayed()]
                        if visible_elements:
                            excel_element = visible_elements[0]
                            found_selector = selector
                            print(f"Found Excel export element using selector: {selector}")
                            break
                except Exception as e:
                    continue
            
            if not excel_element:
                # Try to find export/download sections first
                export_section_selectors = [
                    "//div[contains(@class, 'export')]",
                    "//div[contains(@class, 'download')]",
                    "//section[contains(@class, 'export')]",
                    "//div[contains(text(), 'Export')]",
                    "//div[contains(text(), 'Download')]",
                    "//div[contains(@id, 'export')]",
                    "//div[contains(@id, 'download')]"
                ]
                
                for selector in export_section_selectors:
                    try:
                        export_sections = self.driver.find_elements(By.XPATH, selector)
                        for export_section in export_sections:
                            if export_section.is_displayed():
                                # Look for Excel button within this section
                                try:
                                    excel_element = export_section.find_element(By.XPATH, 
                                        ".//a[contains(text(), 'Excel')] | .//button[contains(text(), 'Excel')] | " +
                                        ".//a[contains(@href, 'excel')] | .//button[contains(@class, 'excel')]")
                                    print(f"Found Excel element in export section")
                                    break
                                except Exception:
                                    continue
                        if excel_element:
                            break
                    except Exception:
                        continue
            
            if excel_element:
                print("Found Excel export button. Attempting to click...")
                
                # Scroll to element to ensure it's visible
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", excel_element)
                time.sleep(1)
                
                # Highlight the element for debugging (optional)
                self.driver.execute_script("arguments[0].style.border='3px solid red'", excel_element)
                time.sleep(1)
                
                # Try different click methods
                click_successful = False
                
                # Method 1: Regular click
                try:
                    excel_element.click()
                    click_successful = True
                    print("Regular click successful")
                except Exception as e:
                    print(f"Regular click failed: {e}")
                
                # Method 2: JavaScript click if regular click fails
                if not click_successful:
                    try:
                        self.driver.execute_script("arguments[0].click();", excel_element)
                        click_successful = True
                        print("JavaScript click successful")
                    except Exception as e:
                        print(f"JavaScript click failed: {e}")
                
                # Method 3: Action chains click
                if not click_successful:
                    try:
                        from selenium.webdriver.common.action_chains import ActionChains
                        actions = ActionChains(self.driver)
                        actions.move_to_element(excel_element).click().perform()
                        click_successful = True
                        print("Action chains click successful")
                    except Exception as e:
                        print(f"Action chains click failed: {e}")
                
                if click_successful:
                    print("Click executed. Waiting for download...")
                    
                    # Wait for download to complete
                    downloaded_file = self.wait_for_download_complete(timeout=30)
                    
                    if downloaded_file:
                        print(f"Excel file successfully downloaded: {downloaded_file}")
                        return True
                    else:
                        print("Download may have started but didn't complete or wasn't detected")
                        return False
                else:
                    print("All click methods failed")
                    return False
                    
            else:
                print("Excel export button not found. Available clickable elements:")
                # Print all clickable elements for debugging
                clickable_elements = self.driver.find_elements(By.XPATH, "//a | //button")
                displayed_elements = [elem for elem in clickable_elements if elem.is_displayed()]
                
                for i, elem in enumerate(displayed_elements[:30]):  # First 30 visible elements
                    try:
                        text = elem.text.strip()
                        tag = elem.tag_name
                        classes = elem.get_attribute('class') or ''
                        href = elem.get_attribute('href') or ''
                        
                        if text or 'export' in classes.lower() or 'download' in classes.lower():
                            print(f"  {i+1}. <{tag}> '{text}' (class: {classes[:50]}) (href: {href[:50]})")
                    except:
                        pass
                return False
                
        except Exception as e:
            print(f"Error during download: {str(e)}")
            return False
        finally:
            if self.driver:
                print("Closing browser...")
                self.driver.quit()

# Usage example
def main():
    url = "https://www.adr.db.com/drwebrebrand/dr-universe/books-open-close"
    
    # Set custom download directory (optional)
    downloads_folder = str(Path.home() / "Downloads")
    
    downloader = DBADRSeleniumDownloader(download_dir=downloads_folder)
    
    print(f"Starting download process...")
    print(f"Files will be saved to: {downloads_folder}")
    
    success = downloader.download_excel_export(url)
    
    if success:
        print("Download completed successfully!")
    else:
        print("Download failed. Check the output above for debugging information.")

if __name__ == "__main__":
    main()
