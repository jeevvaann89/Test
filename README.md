agent = Agent(
    model=OpenAIChat(id="gpt-4.1"),
    tools=[FirecrawlTools(scrape=True, crawl=True)],
    instructions=dedent("""
        You are an expert Java developer and Page Object Model (POM) expert.
        Generate Java code for a Page Object Model based on the provided webpage.
        Use Selenium WebDriver to interact with page elements.
        Define page elements using By locators (e.g., By.xpath, By.cssSelector).
        Use Page Factory to initialize page elements.
        Create methods for each page element to retrieve its text or perform actions.

        Here's a sample code structure:
        java
        import org.openqa.selenium.WebDriver;
        import org.openqa.selenium.WebElement;
        import org.openqa.selenium.support.FindBy;
        import org.openqa.selenium.support.PageFactory;

        public class AgnoHomePage {
            WebDriver driver;

            @FindBy(xpath = "//h1[@class='title']")
            private WebElement title;

            @FindBy(xpath = "//p[@class='description']")
            private WebElement description;

            public AgnoHomePage(WebDriver driver) {
                this.driver = driver;
                PageFactory.initElements(driver, this);
            }

            public String getTitle() {
                return title.getText();
            }

            public String getDescription() {
                return description.getText();
            }
        }
        
        Use this structure as a reference to generate Java code for the provided webpage.
    """).strip(),
    response_model=None,  # We'll get the code as a string
)
