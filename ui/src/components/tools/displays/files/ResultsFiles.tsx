import { useRef } from 'react';
import { Col, Row } from 'react-bootstrap';

// project imports
import { getResultsFile } from '@thorpi/results';
import { useAuth } from '@utilities/auth';
import { ResultRenderProps } from '../../props';

// generic json dump using react-json-view library
const ResultsFiles: React.FC<ResultRenderProps> = ({ result, sha256, tool }) => {
  const { checkCookie } = useAuth();
  const downloadFile = async (sha256: string, tool: string, id: string, fileName: string) => {
    await getResultsFile(sha256, tool, id, fileName, checkCookie).then((res) => {
      if (res.data && res.headers) {
        // turn response data to blob object
        const blob = new Blob([res.data], { type: res.headers['content-type'] });
        // map url to blob in memory
        const url = window.URL.createObjectURL(blob);
        // create anchor tag for blob link
        const link = document.createElement('a');
        // assign href
        link.href = url;
        // set link as download
        link.setAttribute('download', fileName);
        // Append to html link element page
        document.body.appendChild(link);
        // Start download
        link.click();
        // Clean up and remove the link
        link.parentNode?.removeChild(link);
      }
    });
  };
  const filesRef = useRef(null);
  if (result && result.files) {
    return (
      <>
        <Row id={`files_${tool}`} ref={filesRef}>
          <Col className="d-flex justify-content-center">
            <h5>Result Files</h5>
          </Col>
        </Row>
        {result.files &&
          result.files.map((item, idx) => (
            <Row key={idx}>
              <Col className="d-flex justify-content-center">
                <a key={idx} href={`#results-${tool}`} onClick={() => downloadFile(sha256, tool, result.id, item)}>
                  {item}
                </a>
              </Col>
            </Row>
          ))}
      </>
    );
  } else {
    return null;
  }
};

export default ResultsFiles;
