import { useState, useEffect, useRef } from 'react';
import { Button, Card, Col, Row } from 'react-bootstrap';
import { FaAngleDown, FaAngleUp, FaLink } from 'react-icons/fa';
import { InView } from 'react-intersection-observer';
import { toast } from 'react-toastify';
import { ErrorBoundary } from 'react-error-boundary';
import styled from 'styled-components';

// project imports
import Disassembly from './displays/Disassembly';
import Image from './displays/Image';
import JSON from './displays/JSON';
import String from './displays/String';
import Tables from './displays/Tables';
import XML from './displays/XML';
import TC2 from './displays/custom/TC2';
import VBA from './displays/custom/VBA';
import AvMulti from './displays/custom/AvMulti';
import SafeHtml from './SafeHtml';
import Title from '@components/shared/titles/Title';
import RenderErrorAlert from '@components/shared/alerts/RenderErrorAlert';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import Markdown from './displays/Markdown';
import { Output, OutputDisplayType } from '@models/results';

interface ToolResultProps {
  result: Output;
  type: OutputDisplayType;
  header: string;
  sha256: string;
  tool: string;
  updateInView: (inView: boolean, tool: string) => void;
  updateURLSection: (section: string, value: string) => void;
}

const BtnCol = styled(Col)`
  text-align: right;
`;

const ToolResult = ({ result, type, header, sha256, tool, updateInView, updateURLSection }: ToolResultProps) => {
  const [isOpen, setOpened] = useState(false);
  const [scrollRef, setScrollRef] = useState('');
  const [height, setHeight] = useState(0);
  const resultRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // watch for changes to size and update the height
    if (!resultRef.current) return;
    const resizeObserver = new ResizeObserver(() => {
      // Do what you want to do when the size of the element changes
      if (resultRef.current === null) return;
      setHeight(resultRef.current?.clientHeight);
    });
    resizeObserver.observe(resultRef.current);
    return () => resizeObserver.disconnect(); // clean up
  }, []);

  const scrollToFiles = (value: string) => {
    const element = document?.getElementById(value);
    if (element === null) return;
    element.scrollIntoView({ behavior: 'smooth' });
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
    void navigator.clipboard.writeText(window.location.href);
    // notify user that url location was copied to clipboard with toast notification
    const notify = () => toast(`Copied "${window.location.href}" to clipboard!`);
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
        onChange={(inView) => updateInView(inView, tool)}
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
                {result && result.files && type != OutputDisplayType.Image && Object.keys(result.files).length > 0 && (
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
              <BtnCol xs={2}>
                {height >= 85 ? (
                  <Button variant="sm" className="primary-btn mt-1" onClick={() => setOpened(!isOpen)}>
                    {isOpen && <FaAngleUp size={18} />}
                    {!isOpen && <FaAngleDown size={18} />}
                  </Button>
                ) : (
                  <></>
                )}
              </BtnCol>
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
                  {type == OutputDisplayType.Custom && (header == 'symantec' || header == 'clamav') && (
                    <AvMulti result={result} sha256={sha256} tool={tool} />
                  )}
                  {type == OutputDisplayType.Custom && header == 'vbaextraction' && <VBA result={result} />}
                  {type == OutputDisplayType.Custom && (header == 'titanium-core2' || header == 'tc2') && (
                    <TC2 result={result} sha256={sha256} tool={tool} />
                  )}
                  {type == OutputDisplayType.Disassembly && <Disassembly result={result} sha256={sha256} tool={tool} />}
                  {type == OutputDisplayType.Html && <SafeHtml result={result} sha256={sha256} tool={tool} />}
                  {type == OutputDisplayType.Image && <Image result={result} sha256={sha256} tool={tool} />}
                  {type == OutputDisplayType.Json && <JSON result={result} sha256={sha256} tool={tool} />}
                  {type == OutputDisplayType.Markdown && <Markdown result={result} sha256={sha256} tool={tool} />}
                  {type == OutputDisplayType.String && <String result={result} sha256={sha256} tool={tool} errors={[]} warnings={[]} />}
                  {type == OutputDisplayType.Table && <Tables result={result} sha256={sha256} tool={tool} />}
                  {type == OutputDisplayType.Xml && <XML result={result} sha256={sha256} tool={tool} />}
                </Row>
              </div>
            </ErrorBoundary>
          </Card.Body>
        </Card>
      </InView>
    </>
  );
};

export default ToolResult;
