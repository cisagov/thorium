import { useEffect, useState } from 'react';
import { Card, Col, Row } from 'react-bootstrap';
import { JSONTree } from 'react-json-tree';

// project imports
import ResultAlerts from './ResultAlerts';
import { useResultAlerts } from './useResultAlerts';
import { ResultRenderProps } from '../props';
import { OceanJsonTheme, useJsonTreeInvert } from '@components/shared/renderers/jsonTheme';
import { getResultsFile } from '@thorpi/results';
import { useAuth } from '@utilities/auth';

// spec: ../ToolResult.spec.md

// legacy display path — the modular ImageRenderer (detect.ts IMAGE_EXTENSIONS/imageMimeForName) is
// the maintained image renderer; keep this list only until this display is migrated
const SupportedImageFormats = ['png', 'jpeg', 'gif', 'apng', 'avif', 'svg', 'svgz', 'webp'];

/**
 * Result display for image-producing tools: fetches the result's image files, renders them as
 * <img> elements, and shows the accompanying JSON result below. Object URLs are revoked on
 * result change/unmount.
 */
const Image: React.FC<ResultRenderProps> = ({ result, sha256, tool }) => {
  const [images, setImages] = useState<string[]>([]);
  const { errors, warnings, resultsJson, isJson } = useResultAlerts(result.result, false);
  // invert the dark token palette on light-background themes so the tree stays legible
  // const invertTheme = useJsonTreeInvert();

  const { checkCookie } = useAuth();
  // create + revoke object URLs in a single guarded effect so an out-of-order fetch (result changed
  // mid-flight) can't publish stale URLs or revoke ones still bound to a mounted <img>; object URLs
  // live for the document's lifetime until explicitly revoked
  useEffect(() => {
    let active = true;
    const created: string[] = [];
    const fetchFiles = async () => {
      if (result.files === undefined) return;
      for (const fileName of result.files) {
        const extension = fileName.split('.').pop();
        if (!SupportedImageFormats.includes(extension ? extension : '')) continue;
        // get images from the API and build a local URL path for display
        const res = await getResultsFile(sha256, tool, result.id, fileName, () => void checkCookie());
        if (res && res.data) {
          const resultFile = new File([res.data], fileName, {
            type: `image/${extension}`,
          });
          created.push(URL.createObjectURL(resultFile));
        }
      }
      // only publish the URLs if this run is still current; otherwise drop them to avoid a leak
      if (active) setImages(created);
      else created.forEach((url) => URL.revokeObjectURL(url));
    };
    void fetchFiles();
    return () => {
      active = false;
      created.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [result, sha256, tool]);

  return (
    <>
      <Card className="scroll-log tool-result">
        <Row>
          <ResultAlerts errors={errors} warnings={warnings} />
        </Row>
        <center>
          {images.map((image, i) => (
            <Row key={i}>
              <Col>
                <img alt={`${tool} image ${i}`} src={image} />
              </Col>
            </Row>
          ))}
          {isJson && (
            <Row>
              <Col>
                <JSONTree data={resultsJson} shouldExpandNodeInitially={() => true} hideRoot={true} theme={OceanJsonTheme} />
              </Col>
            </Row>
          )}
        </center>
      </Card>
    </>
  );
};

export default Image;
