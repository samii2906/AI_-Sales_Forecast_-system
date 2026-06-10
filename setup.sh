mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = \$PORT\n\
\n\
[theme]\n\
base = \"dark\"\n\
backgroundColor = \"#0e1117\"\n\
secondaryBackgroundColor = \"#1a1d2e\"\n\
primaryColor = \"#b39ddb\"\n\
textColor = \"#e0e0e0\"\n\
" > ~/.streamlit/config.toml