# Adding this app to the shared droplet's Caddy

The droplet's reverse proxy is a colleague's existing Caddy container
(project `lofc`, config at `/opt/lofc/Caddyfile`), not something this repo
runs itself - there's no Caddy/nginx service in `docker-compose.yml` on
purpose, since only one process can own ports 80/443 on a shared host, and
that's already their container.

## What this repo does on its own

`docker-compose.prod.yml` (applied as an extra layer, see its own header
comment) puts the `webapp` service on the colleague's `lofc_default`
Docker network under the fixed alias `match-analysis` - the name their
Caddy config actually references, not our service's own name (`webapp`).
No published port, nothing opened on the host; the droplet's firewall
only allows 22/80/443 inbound regardless.

## What's in the colleague's Caddyfile

Their side (not ours to edit) forwards the registered subdomain straight
to the alias:

```
orient-matchanalysis.duckdns.org {
    reverse_proxy match-analysis:5050
}
```

`match-analysis` resolves because that's the alias our `webapp` service
joins `lofc_default` under (see docker-compose.prod.yml); `5050` is the
port gunicorn binds to inside that container (see `docker-compose.yml`'s
`webapp` service command). Caddy obtains and renews its own Let's Encrypt
certificate for that block automatically - no certbot, no cron, and it
doesn't touch their existing certificate for their own dashboard.

## Sequence to actually go live

1. ✅ DuckDNS subdomain registered: `orient-matchanalysis.duckdns.org`,
   pointed at the droplet's IP.
2. ✅ Sent the colleague the subdomain, this app's public SSH key, the
   internal port (5050), and that we're joining `lofc_default` under the
   alias `match-analysis` (not publishing on 127.0.0.1) - he's confirmed
   the Caddy side is configured to match.
3. On the droplet (once SSH access is set up): `docker compose -f
   docker-compose.yml -f docker-compose.prod.yml up -d` (see
   docker-compose.prod.yml's header).
4. Confirm `https://orient-matchanalysis.duckdns.org` loads and redirects
   to `/login`.
5. Set `FORCE_HTTPS=1` in the droplet's `.env` and restart `webapp` so the
   session cookie is marked Secure (see webapp.py's ProxyFix comment).
