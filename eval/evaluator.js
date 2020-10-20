var {lexer, parser, nodes} = require('sql-parser-mistic');

const express = require('express');
const app = express();
app.use(express.json());
const port = 3000;


function query_notnull(sql, is_list) {
  if (sql == null) {
    return [[], []]
  }

  switch (sql.constructor.name) {
    case "Select":
      var tmp = [[], []];
      var ret;

      tmp = query_notnull([sql.fields, sql.source, sql.joins, sql.unions, sql.order, sql.group, sql.where, sql.limit], is_list);
      tmp[0] = Array.from(new Set(tmp[0]));
      tmp[1] = Array.from(new Set(tmp[1]));

      if (tmp[0].length > 0) {
        if (sql.where == null) {
          sql.where = new nodes.Where(new nodes.Op("=", new nodes.LiteralValue("agg"), new nodes.NumberValue(0)));
        } else{
          sql.where.conditions = new nodes.Op("and", sql.where.conditions, new nodes.Op("=", new nodes.LiteralValue("agg"), new nodes.NumberValue(0)));
        }
        for (var i = 0; i < tmp[0].length; ++i) {
          sql.where.conditions = new nodes.Op("and", sql.where.conditions, new nodes.WhitepaceList([new nodes.LiteralValue(tmp[0][i]), new nodes.LiteralValue("NOT"), new nodes.LiteralValue("NULL")]));
        }
      }

      if (tmp[1].length > 0) {
        for (var i = 0; i < tmp[1].length; ++i) {
          if (is_list[tmp[1][i]] == true) {
            // console.log("to join", tmp[1][i]);
            var join = new nodes.Join(
              new nodes.Table(new nodes.LiteralValue("t_" + tmp[1][i])),
              new nodes.Op("=", new nodes.LiteralValue("id"), new nodes.LiteralValue(new nodes.LiteralValue("t_" + tmp[1][i]), "m_id"))
            );
            if (sql.joins == null) {
              sql.joins = [join];
            } else {
              sql.joins.push(join);
            }
          }
        }
      }
      // console.log(sql);
      // console.log(sql.toString());
      // return empty
      break;
    case "SubSelect":
      query_notnull(sql.select, is_list);
      // return empty
      break;
    case "Join":
      // no join in the initial queryy
      break;
    case "Union":
      query_notnull(sql.query, is_list);
      // return empty
      break;
    case "Intersect":
      query_notnull(sql.query, is_list);
      // return empty
      break;
    case "LiteralValue":
      // console.log("literal", sql.toString(false));
      return [[], [sql.toString(false)]];
      break;
    case "StringValue":
      //return empty
      break;
    case "NumberValue":
      //return empty
      break;
    case "ArgumentListValue":
    case "ListValue":
      return query_notnull(sql.value, is_list);
      break;
    case "WhitepaceList":
      // assume no such expression
      break;
    case "ParameterValue":
      // assume no such expression
      break;
      return query_notnull(sql.value, is_list);
    case "BooleanValue":
      // return empty
      break;
    case "FunctionValue":
      var tmp = query_notnull(sql.arguments, is_list);
      if (sql.udf){
        return tmp;
      } else {
        return [tmp[1], tmp[1]];
      }
      break;
    case "Case":
      var tmp1 = query_notnull(sql.whens, is_list);
      var tmp2 = query_notnull(sql.else, is_list);
      return [tmp1[0].concat(tmp1[0]), tmp1[1].concat(tmp2[1])];
      break;
    case "CaseElse":
      return query_notnull(sql.elseCondition, is_list);
      break;
    case "CaseWhen":
      return query_notnull([sql.whenCondition, sql.resCondition], is_list);
      break;
    case "Order":
      var tmp = query_notnull(sql.orderings, is_list);
      return [tmp[1], tmp[1]];
      break;
    case "OrderArgument":
      return query_notnull(sql.value, is_list);
      break;
    case "Offset":
      // empty
      break;
    case "Limit":
      // empty
      break;
    case "Table":
      // nested query would be SubSelect
      break;
    case "Group":
      var tmp1 = query_notnull(sql.fields, is_list);
      var tmp2 = query_notnull(sql.having, is_list);
      return [tmp1[1].concat(tmp2[0]), tmp1[1].concat(tmp2[1])];
      break;
    case "Where":
      return query_notnull(sql.conditions, is_list);
      break;
    case "Having":
      return query_notnull(sql.conditions, is_list);
      break;
    case "Op":
      var tmp1 = query_notnull(sql.left, is_list);
      var tmp2 = query_notnull(sql.right, is_list);
      if (sql.operation == ">" || sql.operation == "<" || sql.operation == ">=" || sql.operation == "<=") {
        return [tmp1[1].concat(tmp2[1]), tmp1[1].concat(tmp2[1])];
      } else {
        return [tmp1[0].concat(tmp2[0]), tmp1[1].concat(tmp2[1])];
      }
      break;
    case "UnaryOp":
      return query_notnull(sql.operand, is_list);
      break;
    case "BetweenOp":
      return query_notnull(sql.value, is_list);
      break;
    case "Field":
      return query_notnull(sql.field, is_list);
      break;
    case "Star":
      // empty
      break;
    case "Array":
      var tmp = [[], []];
      var ret;
      for (var i = 0; i < sql.length; ++i) {
        ret = query_notnull(sql[i], is_list);
        // console.log("ret", ret);
        tmp[0] = tmp[0].concat(ret[0]);
        tmp[1] = tmp[1].concat(ret[1]);
      }
      // console.log("array", sql.map((x) => {return x ? x.toString() : "";}).join(","), tmp[0], tmp[1]);
      return tmp;
      break;
    default:
      // console.log("default", sql.constructor.name);
      break;
  }
  return [[], []]
}

app.get('/',
  (req, res) => {
    var sql = req.body.sql;
    console.log(sql);
    var is_list = req.body.is_list;

    var lex = lexer.tokenize(sql);
    lex = lex.map(x => {
      if (x[0] == "LITERAL" && x[1].toUpperCase() == "PRESENT_REF") {
        return ["NUMBER", "2014", x[2], x[3]];
      } else {
        return x;
      }
    });
    var parse_tree = parser.parse(lex);
    query_notnull(parse_tree, is_list);

    var r_sql = parse_tree.toString();
    console.log(r_sql);
    console.log();

    res.json(r_sql);
  }
)

app.listen(port, () => console.log(`Evaluator now listening on port ${port}!`))
