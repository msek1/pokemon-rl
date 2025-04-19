FROM alpine

RUN apk add --no-cache git npm nodejs && \
    git clone https://github.com/smogon/pokemon-showdown.git && \
    cd pokemon-showdown && \
    npm install && \
    cp config/config-example.js config/config.js

WORKDIR /pokemon-showdown
ENTRYPOINT ["node", "pokemon-showdown", "start", "--no-security"]
