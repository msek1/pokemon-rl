FROM alpine

RUN <<EOF
apk add git npm
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
EOF

WORKDIR /pokemon-showdown
ENTRYPOINT node pokemon-showdown start --no-security
