module.exports = (q, r, n) => q.headers.token ? n() : r.status(401).end();
