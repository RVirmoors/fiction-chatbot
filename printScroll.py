import time
from os import system
import asyncio
# https://docs.python.org/3/library/asyncio-task.html#running-an-asyncio-program

async def printScroll(fileName):
    with open(fileName) as f:       
        log = f.readlines()
    i = 2
    delay = 12 # START DELAY = 8 seconds for first answer
    while(True):
        if delay > 1:
            delay = delay * 0.75 # delay becomes 9, 6.75, 5...
        system('cls')   # clear screen
        print("".join(log[-i:]))
        i = i+3
        #time.sleep(1)
        await asyncio.sleep(delay)

async def main():
    # print("print-scrollin for 30 seconds ...")
    await asyncio.sleep(2)

    task = asyncio.create_task(printScroll("log.txt"))
    await asyncio.sleep(30)
    task.cancel()

    # system('cls')   # clear screen
    # print("done print-scrollin")

if __name__ == "__main__":
    asyncio.run(main())
