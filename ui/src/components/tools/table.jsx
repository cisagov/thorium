import { useState, useEffect } from 'react';
import { Alert, Card, Table } from 'react-bootstrap';

// project imports
import { getAlerts, ResultsFiles, ChildrenFiles } from '@components';

const JsonTable = ({ results }) => {
  if (results && Array.isArray(results)) {
    return (
      <Table striped="row" hover={true} className="mb-4">
        <tbody>
          {results.map((array, idx) => (
            <tr key={'outer_' + idx}>
              {array.map((entry, innerIdx) =>
                innerIdx == 0 ? (
                  <td key={'inner_' + entry} className="tables-entry-med">
                    {entry}
                  </td>
                ) : (
                  <td key={'inner_' + entry} className="tables-entry-lrg">
                    {entry}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </Table>
    );
  } else {
    return (
      <center>
        <Alert className="mb-2" variant="warning">
          Cannot display result: result is valid JSON, but is not an array of arrays
        </Alert>
      </center>
    );
  }
};

let numLeadHashes = (str) => {
  // Matches one or more # at the start of the string
  const match = str.match(/^#+/);
  // Return the length of the match or 0 if no match
  const count = match ? match[0].length : 0;
  const header = str.slice(count);
  return {
    count: count,
    header: header,
  };
};

const CsvTable = ({ data, name }) => {
  const rows = data.trim().split('\n');
  return (
    <Table striped hover size="sm" className="mb-4 auto-width">
      {rows.length > 0 && (
        <thead>
          <tr>
            {rows[0].split(/,(?=[^\]]*(?:\[|$))/).map((value, fieldIdx) => (
              <th key={`table_${name}_field_0_${fieldIdx}`}>{value.length > 200 ? `${value.substring(0, 600)} ...` : value}</th>
            ))}
          </tr>
        </thead>
      )}
      <tbody>
        {rows.length > 1 &&
          rows.map((row, rowIdx) => (
            <>
              {rowIdx > 0 && (
                <tr key={`table_${name}_row_${rowIdx}`}>
                  {row.split(/,(?=[^\]]*(?:\[|$))/).map((value, fieldIdx) => (
                    <td key={`table_${name}_field_${rowIdx}_${fieldIdx}`}>
                      {value.length > 200 ? `${value.substring(0, 600)} ...` : value}
                    </td>
                  ))}
                </tr>
              )}
            </>
          ))}
      </tbody>
    </Table>
  );
};

const HtmlHeading = ({ heading }) => {
  const { count, header } = numLeadHashes(heading);
  if (count == 1) {
    return <h1>{header}</h1>;
  } else if (count == 2) {
    return <h2>{header}</h2>;
  } else if (count == 3) {
    return <h3>{header}</h3>;
  } else if (count == 4) {
    return <h4>{header}</h4>;
  } else if (count == 5) {
    return <h5>{header}</h5>;
  } else if (count == 6) {
    return <h6>{header}</h6>;
  }
  return <div>{heading}</div>;
};

const splitTableSections = (results) => {
  const rows = results.trim().split('\n');
  let htmlSegments = [];
  let tableRows = '';
  rows.map((row) => {
    if (row === '' || row.startsWith('#') || !row.includes(',')) {
      // header is not a table, reset table row count
      if (tableRows.length > 0) {
        htmlSegments.push((' ' + tableRows).slice(1));
        tableRows = '';
      }
      htmlSegments.push(row);
    } else {
      // Add header and separator
      tableRows += row + '\n';
    }
  });
  if (tableRows.length > 0) {
    htmlSegments.push(tableRows);
  }
  return htmlSegments;
};

const CsvMultiTable = ({ results }) => {
  // split text into rows
  const htmlSegments = splitTableSections(results);
  return (
    <center>
      {htmlSegments.map((segment, idx) => (
        <>
          {segment === '' && <br />}
          {segment.startsWith('#') ? <HtmlHeading heading={segment} /> : <CsvTable data={segment} name={idx} />}
        </>
      ))}
    </center>
  );
};

const Tables = ({ result, sha256, tool }) => {
  const [errors, setErrors] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [resultsJson, setResultsJson] = useState([]);
  const [isJson, setIsJson] = useState(true);

  useEffect(() => {
    // set alerts and process results to json
    getAlerts(result.result, setResultsJson, setWarnings, setErrors, setIsJson, true);
  }, [result]);

  // format string results or ignore result if json
  let parsedResult = '';
  // result is a string, replace new lines and format as such
  if (!isJson) {
    parsedResult = result.result.replace(/\\n/g, '\n').replace(/["]+/g, '');
  } else {
    // ignore the results, they aren't strings
    if (JSON.stringify(resultsJson) == '{}') {
      parsedResult = '';
    } else {
      // there is non-empty json, display as string
      parsedResult = JSON.stringify(resultsJson);
    }
  }

  return (
    <Card className="scroll-log tool-result">
      {errors.map((err, idx) => (
        <center key={idx}>
          <Alert variant="danger">{err}</Alert>
        </center>
      ))}
      {warnings.map((warn, idx) => (
        <center key={idx}>
          <Alert variant="warning">{warn}</Alert>
        </center>
      ))}
      {isJson ? <JsonTable results={parsedResult} /> : <CsvMultiTable results={parsedResult} />}
      <ResultsFiles result={result} sha256={sha256} tool={tool} />
      <ChildrenFiles result={result} tool={tool} />
    </Card>
  );
};

export default Tables;
