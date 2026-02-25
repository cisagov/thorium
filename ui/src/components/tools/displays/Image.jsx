import { useEffect, useState } from 'react';
import { Alert, Card, Col, Row } from 'react-bootstrap';
import { JSONTree } from 'react-json-tree';

// project imports
import { OceanJsonTheme } from './JSON';
import { getAlerts } from '../alerts';
import ResultsFiles from './files/ResultsFiles';
import ChildrenFiles from './files/ChildrenFiles';
import { getResultsFile } from '@thorpi/results';
import { useAuth } from '@utilities/auth';

const SupportedImageFormats = ['png', 'jpeg', 'gif', 'apng', 'avif', 'svg', 'svgz', 'webp'];

const Image = ({ result, sha256, tool }) => {
  const [images, setImages] = useState([]);
  const [errors, setErrors] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [resultsJson, setResultsJson] = useState({});
  const [isJson, setIsJson] = useState(true);

  const { checkCookie } = useAuth();
  useEffect(() => {
    const fetchFiles = async () => {
      const fileData = [];
      if (!('files' in result)) return;
      for (const fileName of result.files) {
        const extension = fileName.split('.').pop();
        if (!SupportedImageFormats.includes(extension)) continue;
        // get images from the API and build a local URL path for display
        const res = await getResultsFile(sha256, tool, result.id, fileName, checkCookie);
        if (res && res.data) {
          const resultFile = new File([res.data], fileName, {
            type: `image/${extension}`,
          });
          fileData.push(URL.createObjectURL(resultFile));
        }
      }
      // set the built image URLs into a list
      setImages(fileData);
    };
    fetchFiles();
    // set alerts and process results to json
    getAlerts(result.result, setResultsJson, setWarnings, setErrors, setIsJson);
  }, [result, sha256, tool]);

  return (
    <>
      <Card className="scroll-log tool-result">
        <Row>
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
        </Row>
        <center>
          {images.map((image, i) => (
            <Row key={i}>
              <Col>
                <img alt={`${tool} image ${i}`} src={image} />
              </Col>
            </Row>
          ))}
          {isJson && Object.keys(resultsJson).length > 0 && (
            <Row>
              <Col>
                <JSONTree
                  data={resultsJson}
                  shouldExpandNodeInitially={() => true}
                  hideRoot={true}
                  theme={OceanJsonTheme}
                  invertTheme={false}
                />
              </Col>
            </Row>
          )}
        </center>
        <ResultsFiles result={result} sha256={sha256} tool={tool} />
        <ChildrenFiles result={result} tool={tool} />
      </Card>
    </>
  );
};

export default Image;
