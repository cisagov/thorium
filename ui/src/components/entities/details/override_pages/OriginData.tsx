import { Fragment } from 'react';
import { Col, Row } from 'react-bootstrap';

// project imports
import Subtitle from '@components/shared/titles/Subtitle';
import { Origin } from '@models/entities';

interface OriginDataProps {
  origin: Origin;
}

const OriginData = ({ origin }: OriginDataProps) => {
  const originType = Object.keys(origin)[0];
  const originData = (origin as Record<string, Record<string, unknown>>)[originType];
  return (
    <>
      <Row>
        <Col className="origin-field-name" xs={2}>
          <Subtitle>Type</Subtitle>
        </Col>
        <Col>
          <p>{originType}</p>
        </Col>
      </Row>
      {originData &&
        typeof originData === 'object' &&
        Object.keys(originData).map((key) => {
          if (key == 'carved_origin') {
            const carvedOrigin = originData[key] as string | Record<string, Record<string, string>>;
            return (
              <Fragment key={key}>
                <br />
                {carvedOrigin == 'Unknown' && (
                  <Row>
                    <Col className="origin-field-name" xs={2}>
                      <Subtitle>Carved Type</Subtitle>
                    </Col>
                    <Col>
                      <p>{carvedOrigin}</p>
                    </Col>
                  </Row>
                )}
                {carvedOrigin != 'Unknown' && typeof carvedOrigin === 'object' && (
                  <>
                    <Row>
                      <Col className="origin-field-name" xs={2}>
                        <Subtitle>Carved Type</Subtitle>
                      </Col>
                      <Col>
                        <p>{Object.keys(carvedOrigin)[0]}</p>
                      </Col>
                    </Row>
                    {Object.keys(carvedOrigin[Object.keys(carvedOrigin)[0]]).map((carvedKey) => (
                      <Row key={carvedKey}>
                        <Col className="origin-field-name" xs={2}>
                          <Subtitle>{carvedKey}</Subtitle>
                        </Col>
                        <Col>
                          <p>{carvedOrigin[Object.keys(carvedOrigin)[0]][carvedKey]}</p>
                        </Col>
                      </Row>
                    ))}
                  </>
                )}
              </Fragment>
            );
          } else {
            return (
              <Row key={key}>
                {originData[key] != null && originData[key] != '' && (
                  <Col className="origin-field-name" xs={2}>
                    <Subtitle>{key}</Subtitle>
                  </Col>
                )}
                {originData[key] != null && originData[key] != '' && key == 'parent' && (
                  <Col>
                    <a className="origin-sha256" href={`/file/${originData[key] as string}`}>
                      {originData[key] as string}
                    </a>
                    <a className="short-origin-sha256" href={`/file/${originData[key] as string}`}>
                      {(originData[key] as string).substring(0, 20) + '...'}
                    </a>
                  </Col>
                )}
                {key != 'parent' && (
                  <Col>
                    <p>{String(originData[key])}</p>
                  </Col>
                )}
              </Row>
            );
          }
        })}
    </>
  );
};

export default OriginData;
