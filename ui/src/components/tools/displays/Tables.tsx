import { Card, Table } from 'react-bootstrap';

// project imports
import ResultAlerts from './ResultAlerts';
import { useResultAlerts } from './useResultAlerts';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { formatResultBody } from '../alerts';
import { ResultRenderProps } from '../props';

// spec: ../ToolResult.spec.md

/** Render a JSON array-of-arrays result as a striped table, or a warning if the shape is unexpected. */
const JsonTable = ({ results }: { results: string }) => {
  if (results && Array.isArray(results)) {
    return (
      <Table striped="row" hover={true} className="mb-4">
        <tbody>
          {(results as string[][]).map((array: string[], idx: number) => (
            <tr key={'outer_' + idx}>
              {array.map((entry: string, innerIdx: number) =>
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
      <AlertBanner severity={Severity.Warning}>Cannot display result: result is valid JSON, but is not an array of arrays</AlertBanner>
    );
  }
};

/**
 * Parse a Markdown-style heading string into its level and text.
 *
 * @param str - A line that may begin with one or more leading `#` characters.
 * @returns The number of leading `#` (`count`) and the remaining text (`header`).
 */
export const numLeadHashes = (str: string) => {
  const match = str.match(/^#+/);
  const count = match ? match[0].length : 0;
  const header = str.slice(count);
  return {
    count: count,
    header: header,
  };
};

/** Render a single CSV segment as a table, treating the first row as the header. */
const CsvTable = ({ data, name }: { data: string; name: string | number }) => {
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

/** Render a `#`-prefixed line as the matching HTML heading level, falling back to a plain div. */
const HtmlHeading = ({ heading }: { heading: string }) => {
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

/**
 * Split CSV-with-headings text into ordered segments, grouping contiguous comma-separated data rows
 * into single table segments and emitting blank lines and `#` headings as their own segments.
 *
 * @param results - The raw text to split (heading lines, blank lines, and CSV rows interleaved).
 * @returns The segments in source order; each is either a heading/blank line or a table block.
 */
export const splitTableSections = (results: string): string[] => {
  const rows = results.trim().split('\n');
  const htmlSegments: string[] = [];
  let tableRows = '';
  rows.map((row) => {
    if (row === '' || row.startsWith('#') || !row.includes(',')) {
      // a non-table line closes any open table block before it is emitted on its own
      if (tableRows.length > 0) {
        htmlSegments.push((' ' + tableRows).slice(1));
        tableRows = '';
      }
      htmlSegments.push(row);
    } else {
      tableRows += row + '\n';
    }
  });
  if (tableRows.length > 0) {
    htmlSegments.push(tableRows);
  }
  return htmlSegments;
};

/** Render CSV text containing headings and multiple tables as a stacked sequence of tables/headings. */
const CsvMultiTable = ({ results }: { results: string }) => {
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

/** Render a tool result as a table: a JSON array-of-arrays, or CSV text with optional headings. */
const Tables: React.FC<ResultRenderProps> = ({ result }) => {
  const { errors, warnings, resultsJson, isJson } = useResultAlerts(result.result, true, []);
  const parsedResult = formatResultBody(result.result, isJson, resultsJson);

  return (
    <Card className="scroll-log tool-result">
      <ResultAlerts errors={errors} warnings={warnings} />
      {isJson ? <JsonTable results={parsedResult} /> : <CsvMultiTable results={parsedResult} />}
    </Card>
  );
};

export default Tables;
