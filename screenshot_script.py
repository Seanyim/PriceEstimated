import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Using 1440x900 for a good dashboard view
        context = await browser.new_context(viewport={'width': 1440, 'height': 900}, device_scale_factor=2)
        page = await context.new_page()
        
        print("Navigating to app...")
        await page.goto("http://localhost:8501")
        
        print("Waiting for app to load...")
        # Wait for the main title
        await page.wait_for_selector("text=企业估值系统", timeout=30000)
        await page.wait_for_timeout(3000)
        
        print("Taking Dashboard screenshot...")
        try:
            # Click on trend analysis tab
            await page.locator("button:has-text('📈 趋势分析')").click()
            await page.wait_for_timeout(3000)
            # Hide the sidebar if possible for a cleaner screenshot, or leave it.
            # Streamlit sidebar can be collapsed.
            # await page.locator("[data-testid='stSidebarCollapseButton']").click()
            # await page.wait_for_timeout(1000)
        except Exception as e:
            print("Could not click on Trend Analysis:", e)
            
        await page.screenshot(path="assets/images/dashboard_placeholder.png")
        print("Saved dashboard_placeholder.png")
        
        print("Taking Valuation screenshot...")
        try:
            await page.locator("button:has-text('🧮 估值模型')").click()
            await page.wait_for_timeout(2000)
            
            await page.locator("button:has-text('🚀 DCF 估值')").click()
            await page.wait_for_timeout(3000)
        except Exception as e:
            print("Could not click on Valuation Models:", e)
            
        await page.screenshot(path="assets/images/valuation_placeholder.png")
        print("Saved valuation_placeholder.png")
        
        await browser.close()
        print("Done.")

if __name__ == "__main__":
    asyncio.run(run())
