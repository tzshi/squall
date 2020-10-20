spawn = require('cross-spawn')
fs = require('fs')
UglifyJS = require('uglify-js')
require('coffee-script/register')
pkg = require('./package.json')

header = """
/*!
 * SQLParser #{pkg.version}
 * Copyright 2012-2015 Andy Kent <andy@forward.co.uk>
 * Copyright 2015-2018 Damien "Mistic" Sorel (https://www.strangeplanet.fr)
 * Licensed under MIT (http://opensource.org/licenses/MIT)
 */
"""

run = (args, cb) ->
  proc =         spawn './node_modules/.bin/coffee', args
  proc.stderr.on 'data', (buffer) -> console.log buffer.toString()
  proc.on        'exit', (status) ->
    process.exit(1) if status != 0
    cb() if typeof cb is 'function'

build = (cb) ->
  files = fs.readdirSync 'src'
  files = ('src/' + file for file in files when file.match(/\.coffee$/))
  run ['-c', '-o', 'lib'].concat(files), cb

task 'build', 'Run full build', ->
  invoke 'build:compile'
  invoke 'build:parser'
  setTimeout (-> invoke 'build:browser'), 100

task 'build:compile', 'Compile all coffee files to js',
  build

task 'build:parser', 'rebuild the Jison parser', ->
  parser = require('./src/grammar').parser
  fs.writeFileSync 'lib/compiled_parser.js', parser.generate()

task 'build:browser', 'Build a single JS file suitable for use in the browser', ->
  code = ''
  for name in ['lexer', 'compiled_parser', 'nodes', 'parser', 'sql_parser']
    code += """
      require['./#{name}'] = new function() {
        var exports = this;
        #{fs.readFileSync "lib/#{name}.js"}
      };
    """
  code = """
    #{header}
    (function(root) {
      var SQLParser = function() {
        function require(path){ return require[path]; }
        #{code}
        return require['./sql_parser']
      }();

      if(typeof define === 'function' && define.amd) {
        define(function() { return SQLParser });
      } else { root.SQLParser = SQLParser }
    }(this));
  """
  fs.writeFileSync './browser/sql-parser.js', code

  invoke 'build:minify'

task 'build:minify', 'Minify the builded JS file suitable for use in the browser', ->
  minified = UglifyJS.minify './browser/sql-parser.js'

  code = """
    #{header}
    #{minified.code}
  """

  fs.writeFileSync './browser/sql-parser.min.js', code


