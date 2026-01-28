import asyncio
import time

import aiohttp

URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "test_image.jpg"
REQUESTS = 100


async def send_request(session, idx):
    with open(IMAGE_PATH, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("file", f, filename="test.jpg", content_type="image/jpeg")

        start_time = time.perf_counter()
        async with session.post(URL, data=data) as response:
            elapsed = time.perf_counter() - start_time

            try:
                body = await response.json()
            except Exception:
                body = await response.text()

            return {
                "request": idx,
                "status": response.status,
                "body": body,
                "time_sec": round(elapsed, 3),
            }

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i + 1) for i in range(REQUESTS)]
        results = await asyncio.gather(*tasks)

    # Print nicely
    for r in results:
        print(
            f"Request {r['request']:02d} | "
            f"Status: {r['status']} | "
            f"Time: {r['time_sec']}s | "
            #f"Response: {r['body']}"
        )

if __name__ == "__main__":
    asyncio.run(main())
