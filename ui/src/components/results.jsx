import React, { useEffect, useState } from 'react';
import { getResults } from '@thorpi';
import { Alert, Col, Row, Container } from 'react-bootstrap';

// project imports
import { LoadingSpinner, Tool } from '@components';
import { useAuth, updateURLSection, scrollToSection } from '@utilities';

const Results = ({ sha256, results, setResults, numResults, setNumResults }) => {
  // whether content is currently loading
  const [loading, setLoading] = useState(false);
  const [inViewElements, setInViewElements] = useState([]);
  const { checkCookie } = useAuth();
  // get results from API
  useEffect(() => {
    let isSubscribed = true;
    const fetchData = async () => {
      setLoading(true);
      const resultsRes = await getResults(sha256, checkCookie, {});
      // results must be set and
      // there can only be one outstanding subscribed request
      if (resultsRes && 'results' in resultsRes && isSubscribed) {
        // pass back number of results to parent
        setNumResults(Object.keys(resultsRes.results).length);
        setResults(resultsRes.results);
      }
      setLoading(false);
    };
    fetchData();
    return () => {
      isSubscribed = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sha256]);

  // update whether an element is in the view port
  const updateInView = (inView, entry) => {
    if (inView) {
      setInViewElements((previousInViewElements) => [...previousInViewElements, entry].sort());
    } else {
      setInViewElements((previousInViewElements) => {
        return previousInViewElements.filter((element) => element != entry).sort();
      });
    }
  };

  // remove hidden display typed results from results object
  Object.keys(results)
    .sort()
    .map((image) => {
      if (results[image][0]['display_type'] && results[image][0]['display_type'] == 'Hidden') {
        delete results[image];
      }
    });

  // floating table of contents object
  const ResultsTableOfContents = ({ results }) => {
    return (
      <nav className="results-toc">
        <ul className="ul no-bullets">
          {Object.keys(results)
            .sort()
            .map((image) => (
              <li key={`results-${image}-toc`} className="results-toc-item">
                <a
                  href={`#results-${image}`}
                  onClick={() => scrollToSection(`results-tab-${image}`)}
                  className={`${inViewElements.includes(image) ? 'selected' : 'unselected'}`}
                >
                  {image}
                </a>
                <hr className="m-1" />
              </li>
            ))}
        </ul>
      </nav>
    );
  };

  return (
    <div id="results-tab" className="navbar-scroll-offset results-container">
      <LoadingSpinner loading={loading}></LoadingSpinner>
      {results && typeof results === 'object' && !loading && (
        <>
          <div className="results-col">
            {numResults == 0 && !loading && (
              <>
                <br />
                <Alert variant="" className="info">
                  <Alert.Heading>
                    <center>
                      <h3>No Tool Results Available</h3>
                    </center>
                  </Alert.Heading>
                  <center>
                    <p>Check back later for updated results</p>
                  </center>
                </Alert>
              </>
            )}
            {Object.keys(results)
              .sort()
              .map((image) => (
                <Tool
                  key={image}
                  header={image}
                  type={results[image][0]['display_type'] ? results[image][0]['display_type'] : 'Json'}
                  tool={image}
                  sha256={sha256}
                  updateInView={updateInView}
                  updateURLSection={updateURLSection}
                  result={results[image][0]}
                />
              ))}
          </div>
          {Object.keys(results).length > 0 && (
            <div className="results-toc-col">
              <ResultsTableOfContents results={results} />
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Results;
