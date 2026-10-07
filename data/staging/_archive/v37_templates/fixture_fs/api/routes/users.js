const db = require('../db');
module.exports.list = (q, r) => r.json(db.users());
