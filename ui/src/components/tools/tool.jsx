import React, { useState, useEffect, useRef } from 'react';
import { Button, Card, Col, Row } from 'react-bootstrap';
import { FaAngleDown, FaAngleUp, FaLink } from 'react-icons/fa';
import { InView } from 'react-intersection-observer';
import { toast } from 'react-toastify';
import { ErrorBoundary } from 'react-error-boundary';

// project imports
import { Disassembly, Image, Json, OverlayTipRight, String, Tables, Title, SafeHtml, Markdown, Xml } from '@components';
import Tc2 from './custom/tc2';
import VBA from './custom/vba';
import AvMulti from './custom/avmulti';
import { RenderErrorAlert } from '@components';

const Tool = ({ result, type, header, sha256, tool, updateInView, updateURLSection }) => {
  const [isOpen, setOpened] = useState(false);
  const [scrollRef, setScrollRef] = useState('');
  const [height, setHeight] = useState(0);
  const resultRef = useRef();

  useEffect(() => {
    // watch for changes to size and update the height
    if (!resultRef.current) return;
    const resizeObserver = new ResizeObserver(() => {
      // Do what you want to do when the size of the element changes
      setHeight(resultRef.current.clientHeight);
    });
    resizeObserver.observe(resultRef.current);
    return () => resizeObserver.disconnect(); // clean up
  }, []);

  const scrollToFiles = (value) => {
    document.getElementById(value).scrollIntoView({ behavior: 'smooth' });
  };

  // when the scroll ref changes, jump to ref
  useEffect(() => {
    if (scrollRef != '') {
      scrollToFiles(scrollRef);
      setScrollRef('');
    }
  }, [scrollRef]);

  const updateSelectedResultsSection = () => {
    // update url location with selected results section
    updateURLSection('results', `${tool}`);
    // copy url with updated section to clipboard
    navigator.clipboard.writeText(window.location);
    // notify user that url location was copied to clipboard with toast notification
    const notify = () => toast(`Copied "${window.location}" to clipboard!`);
    notify();
  };

  return (
    <>
      <InView
        as="div"
        id={`results-tab-${tool}`}
        className="navbar-scroll-offset"
        rootMargin="-60px 0px 0px 0px"
        threshold={isOpen ? 0 : 0.33}
        root={document.querySelector('results-tab')}
        onChange={(inView, entry) => updateInView(inView, tool)}
      >
        <Card className="tool-card mt-2 results-content" ref={resultRef}>
          <Card.Header className="py-2">
            <Row className="my-0">
              <Col xs={2}>
                {result && result.children && Object.keys(result.children).length > 0 && (
                  <OverlayTipRight tip={`Click to jump to children`}>
                    <div
                      className="general-tag tag-item clickable m-1"
                      onClick={() => {
                        setOpened(true);
                        setScrollRef(`children_${tool}`);
                      }}
                    >
                      {`${Object.keys(result.children).length}
                      ${Object.keys(result.children).length == 1 ? 'Child' : 'Children'}`}
                    </div>
                  </OverlayTipRight>
                )}
                {result && result.files && type != 'Image' && Object.keys(result.files).length > 0 && (
                  <OverlayTipRight tip={`Click to jump to result file(s)`}>
                    <div
                      className="general-tag tag-item clickable mt-2"
                      onClick={() => {
                        setOpened(true);
                        setScrollRef(`files_${tool}`);
                      }}
                    >
                      {`${Object.keys(result.files).length}
                      ${Object.keys(result.children).length == 1 ? 'File' : 'Files'}`}
                    </div>
                  </OverlayTipRight>
                )}
              </Col>
              <Col className="d-flex justify-content-center">
                <a className="title-link" onClick={() => updateSelectedResultsSection()}>
                  <Title small>
                    <FaLink className="title-link-no-color me-3" size={12} />
                    {header}
                  </Title>
                </a>
              </Col>
              <Col xs={2} className="btn-align">
                {height >= 85 ? (
                  <Button variant="sm" className="primary-btn mt-1" onClick={() => setOpened(!isOpen)}>
                    {isOpen && <FaAngleUp size={18} />}
                    {!isOpen && <FaAngleDown size={18} />}
                  </Button>
                ) : (
                  <></>
                )}
              </Col>
            </Row>
          </Card.Header>
          <Card.Body>
            <ErrorBoundary
              fallback={
                <RenderErrorAlert
                  message={
                    'Uh Oh! An error occurred while rendering this result, please report it to your Thorium admins.\nNote: This may be caused by an image with a misconfigured display_type. '
                  }
                />
              }
            >
              <div className={isOpen ? '' : 'collapsed'}>
                <Row className="d-flex justify-content-center">
                  {type == 'Custom' && (header == 'symantec' || header == 'clamav') && <AvMulti result={result} />}
                  {type == 'Custom' && header == 'vbaextraction' && <VBA result={result} />}
                  {type == 'Custom' && (header == 'titanium-core2' || header == 'tc2') && <Tc2 result={result} />}
                  {type == 'Disassembly' && <Disassembly result={result} sha256={sha256} tool={tool} />}
                  {type == 'Hidden' && false && <div>Hide this result</div>}
                  {(type == 'Html' || type == 'HTML') && <SafeHtml result={result} sha256={sha256} tool={tool} />}
                  {type == 'Image' && <Image result={result} sha256={sha256} tool={tool} />}
                  {(type == 'Json' || type == 'JSON') && <Json result={result} sha256={sha256} tool={tool} />}
                  {type == 'Markdown' && <Markdown result={result} sha256={sha256} tool={tool} />}
                  {type == 'String' && <String result={result} sha256={sha256} tool={tool} errors={[]} warnings={[]} />}
                  {type == 'Table' && <Tables result={result} sha256={sha256} tool={tool} />}
                  {type == 'Xml' || (type == 'XML' && <Xml result={result} sha256={sha256} tool={tool} />)}
                </Row>
              </div>
            </ErrorBoundary>
          </Card.Body>
        </Card>
      </InView>
    </>
  );
};

export default Tool;
