## Download some data

I propose you put your data in `/data/dlst`

If you want to put it somewhere else, you can override this by setting the env var
`DLST_DATA_DIR` when you run the code.

```
sudo mkdir -p /data
sudo chown -R $USER:$USER /data
```

Then fetch the data, put it there and uncompress it

```
wget 'https://drop.too.gy/066e0bea-af2b-702e-8000-f91c9f0c44ce-dlsttar.tar.bz2'
mv 066e0bea-af2b-702e-8000-f91c9f0c44ce-dlsttar.tar.bz2 /data/
cd /data
tar xvf 066e0bea-af2b-702e-8000-f91c9f0c44ce-dlsttar.tar.bz2
rm 066e0bea-af2b-702e-8000-f91c9f0c44ce-dlsttar.tar.bz2
```

You should have the following:

```
/data/dlst ❯ tree
.
├── paristemp
│   ├── daily.pq
│   └── hourly.pq
└── sp500
    ├── sp500_companies.csv
    ├── sp500_index.pq
    └── sp500_stocks.pq

3 directories, 5 files
```

## Setting up your Python env using uv...

See instructions for setting up uv here: 

https://docs.astral.sh/uv/getting-started/installation/

Or if you're on Linux / macOS, simply run

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

And add `~/.cargo/bin` to your `$PATH`.

Then `cd` in the root of this repo and run

```
uv sync
```

This will create an env in `.venv`, which you can activate with

```
source .venv/bin/activate
```

Personally, I have the following in my `~/.zshrc` to automatically activate my Python
envs whenver I `cd` into a directory that has a `.venv` subdirectory

```
autoload -U add-zsh-hook
activate_if_env() {
    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
    fi
}
add-zsh-hook chpwd activate_if_env
activate_if_env
```
