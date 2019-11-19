git clone https://github.com/huggingface/transformers ${1}
cd ${1}
git fetch origin 74d0bcb6ff692dbaa52da1fdc2b80ece06f5fbe5 --depth=1
git reset --hard FETCH_HEAD

