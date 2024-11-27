---
title: News Caster
emoji: 🏢
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
---

## News Caster

Thinking of consolidating news for the day and be able to get the agent to answer questions about the news.

Will also start implementing a tool to create a way to turn all summarize news in a podcast audio.

### Running it locally
Start by running `./build-docker` and copy a copy of the `sample.env` file to `.env`.
You would first need to start the database and initialize it.
`docker compose up db -d`
This is start the database in the background.
Then you will need to run `./init-db.sh` to initialize the database with langfuse related details.

Once the above is done, you should be able to run `./start-docker` to start everything up and play with the host by running `python agent.py`.

### If any issues
If you are having issues, please open an issue on the [GitHub repo](https://github.com/daniel-ltw/news-caster/issues)
