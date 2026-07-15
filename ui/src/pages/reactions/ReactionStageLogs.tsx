import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { ButtonToolbar, ButtonGroup, Button, Card } from 'react-bootstrap';
import { FaAngleDown, FaAngleDoubleDown, FaAngleDoubleUp, FaAngleUp } from 'react-icons/fa';

// project imports
import Page from '@components/pages/Page';
import { OverlayTipBottom } from '@components/shared/overlay/tips';
import { useAuth } from '@utilities/auth';
import { getReactionStageLogs } from '@thorpi/reactions';
import { AnsiText } from './AnsiText';

const ReactionStageLogs = () => {
  const { reactionID, group, stage } = useParams<{ reactionID: string; group: string; stage: string }>();
  const [cursor, setCursor] = useState(0);
  const [startLogLine, setStartLogLine] = useState(0);
  const [endLogLine, setEndLogLine] = useState(0);
  const [tailLogs, setTailLogs] = useState(true);
  const [pagingDown, setPagingDown] = useState(false);
  const [reactionStageLogs, setReactionStageLogs] = useState<string[]>([]);
  const [logUpdateTimeout, setLogUpdateTimeout] = useState(10);
  const { checkCookie } = useAuth();
  const maxRenderedLines = 1000;
  const limit = 1000;

  // fetch a page of logs starting at the cursor
  const getLogPage = async () => {
    const stageLogs = await getReactionStageLogs(
      group!,
      reactionID!,
      stage!,
      () => {
        void checkCookie();
      },
      cursor as unknown as null,
      limit,
    );

    if (!stageLogs) {
      setLogUpdateTimeout(10000);
      return;
    }

    if (stageLogs.length == 0) {
      setTailLogs(true);
      setLogUpdateTimeout(10000);
      return;
    } else {
      setLogUpdateTimeout(100);
    }

    let logs: string[] = [];
    if (startLogLine > cursor && startLogLine - cursor > endLogLine - startLogLine) {
      setStartLogLine(cursor);
      setEndLogLine(cursor + stageLogs.length);
      logs = stageLogs;
    } else if (startLogLine > cursor) {
      const endLogBufferIndex = reactionStageLogs.length - (startLogLine - cursor);
      logs = stageLogs.concat(reactionStageLogs.slice(0, endLogBufferIndex));
      setStartLogLine(cursor);
      setEndLogLine(cursor + logs.length);
    } else {
      let logStart = startLogLine;
      if (endLogLine + stageLogs.length - startLogLine > maxRenderedLines) {
        logStart = endLogLine + stageLogs.length - maxRenderedLines;
      }
      let startLogBufferIndex = 0;
      if (logStart > 0) {
        startLogBufferIndex = stageLogs.length;
      }
      logs = reactionStageLogs.slice(startLogBufferIndex).concat(stageLogs);
      setStartLogLine(logStart);
      setEndLogLine(endLogLine + stageLogs.length);
    }
    setReactionStageLogs(logs);
  };

  const pageToEndAndFollow = () => {
    setTailLogs(true);
    setPagingDown(true);
    setCursor(endLogLine);
  };

  const pageDown = () => {
    setTailLogs(false);
    setPagingDown(true);
    setCursor(endLogLine);
  };

  const pageToTop = () => {
    setTailLogs(false);
    setPagingDown(false);
    setCursor(0);
  };

  const pageUp = () => {
    setTailLogs(false);
    setPagingDown(false);
    setCursor(startLogLine - limit < 0 ? 0 : startLogLine - limit);
  };

  // get first page on mount
  useEffect(() => {
    void getLogPage();
  }, []);

  // auto-refresh logs when tailing
  useEffect(() => {
    const intervalId = setInterval(() => {
      if (tailLogs) {
        if (endLogLine != cursor) {
          setCursor(endLogLine);
        } else {
          void getLogPage();
        }
      }
    }, logUpdateTimeout);
    return () => clearInterval(intervalId);
  }, [tailLogs, endLogLine, logUpdateTimeout]);

  // fetch page when cursor changes
  useEffect(() => {
    if (cursor < startLogLine || cursor >= endLogLine) {
      void getLogPage();
    }
  }, [cursor]);

  // scroll to correct end when logs update
  const startLogsRef = useRef<HTMLDivElement>(null);
  const endLogsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (pagingDown) {
      endLogsRef.current?.scrollIntoView({ block: 'end', behavior: 'auto' });
    } else {
      startLogsRef.current?.scrollIntoView({ block: 'start', behavior: 'auto' });
    }
  }, [reactionStageLogs, pagingDown]);

  return (
    <Page title="Reaction Stage Logs">
      <br />
      <Card className="log-box panel">
        <Card.Header>
          <ButtonToolbar className="d-flex justify-content-center">
            <ButtonGroup>
              <Button variant="" onClick={pageToTop} className="log-nav-button primary-btn">
                <OverlayTipBottom tip={'Scroll to start of logs'}>
                  <FaAngleDoubleUp size={18} />
                </OverlayTipBottom>
              </Button>
              <Button variant="" onClick={pageUp} className="log-nav-button primary-btn">
                <OverlayTipBottom tip={'Scroll up in logs'}>
                  <FaAngleUp size={18} />
                </OverlayTipBottom>
              </Button>
              <Button variant="" onClick={pageDown} className="log-nav-button primary-btn">
                <OverlayTipBottom tip={'Scroll down in logs'}>
                  <FaAngleDown size={18} />
                </OverlayTipBottom>
              </Button>
              <Button variant="" onClick={pageToEndAndFollow} className="log-nav-button primary-btn">
                <OverlayTipBottom tip={'Scroll to end of logs and follow'}>
                  <FaAngleDoubleDown size={18} />
                </OverlayTipBottom>
              </Button>
            </ButtonGroup>
          </ButtonToolbar>
        </Card.Header>
        <Card.Body className="scrollable-card">
          {reactionStageLogs &&
            reactionStageLogs.map((line, idx) => (
              <div key={startLogLine + idx} className="raw-log-line" ref={endLogsRef}>
                <i className="log-line-index secondary-text">{startLogLine + idx}&emsp;</i>
                <AnsiText text={line}></AnsiText>
              </div>
            ))}
        </Card.Body>
      </Card>
    </Page>
  );
};

export default ReactionStageLogs;
