agent = Agent(
    model=OpenAIChat(id="gpt-4.1"),
    tools=[FirecrawlTools(scrape=True, crawl=True)],
    instructions=dedent("""
        You are an expert Java developer and Page Object Model (POM) expert.
        Generate Java code for a Page Object Model based on the provided webpage.
        Use Selenium WebDriver to interact with page elements.
        Define page elements using By locators (e.g., By.xpath, By.cssSelector).
        Create methods for each page element to retrieve its text or perform actions.

        Here's a sample code structure:
        ```
        public class AgnoHomePage {
            private WebDriver driver;
            private By title = By.xpath("//h1[@class='title']");
            private By description = By.xpath("//p[@class='description']");

            public AgnoHomePage(WebDriver driver) {
                this.driver = driver;
            }

            public String getTitle() {
                return driver.findElement(title).getText();
            }

            public String getDescription() {
                return driver.findElement(description).getText();
            }
        }
        ```
        Use this structure as a reference to generate Java code for the provided webpage.
    """).strip(),
    response_model=None, # We'll get the code as a string
)
