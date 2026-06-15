import React, { useEffect, useState, useMemo } from 'react';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';

// project imports
const ToolResult = React.lazy(() => import('@components/tools/ToolResult'));
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { useAuth } from '@utilities/auth';
import { updateURLSection } from '@utilities/url';
import { scrollToSection } from '@utilities/interactions';
import { getResults } from '@thorpi/results';
import { OutputDisplayType, type Output } from '@models/results';

// spec: ./files.spec.md

type ParsedResults = Record<string, Output[]>;

interface ResultsTableOfContentsProps {
  parsedResults: ParsedResults;
  inViewElements: string[];
}

const ResultsTableOfContents = ({ parsedResults, inViewElements }: ResultsTableOfContentsProps) => {
  return (
    <nav className="results-toc">
      <ul className="ul no-bullets">
        {parsedResults &&
          typeof parsedResults === 'object' &&
          Object.keys(parsedResults)
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

interface ResultsProps {
  sha256: string;
  results: ParsedResults;
  setResults: (results: ParsedResults) => void;
  numResults: number;
  setNumResults: (num: number) => void;
  allowHashUpdate?: boolean;
}

const Results = ({ sha256, results, setResults, numResults, setNumResults }: ResultsProps) => {
  const [loading, setLoading] = useState(false);
  const [inViewElements, setInViewElements] = useState<string[]>([]);
  const { checkCookie } = useAuth();

  // get results from API
  useEffect(() => {
    let isSubscribed = true;
    const fetchData = async () => {
      setLoading(true);
      const resultsRes = await getResults(
        sha256,
        () => {
          void checkCookie();
        },
        {},
      );
      if (resultsRes && 'results' in resultsRes && isSubscribed) {
        setNumResults(Object.keys(resultsRes.results).length);
        setResults(resultsRes.results);
      }
      setLoading(false);
    };
    void fetchData();
    return () => {
      isSubscribed = false;
    };
  }, [sha256]);

  // update whether an element is in the viewport
  const updateInView = (inView: boolean, entry: string) => {
    if (inView) {
      setInViewElements((prev) => [...prev, entry].sort());
    } else {
      setInViewElements((prev) => prev.filter((element) => element != entry).sort());
    }
  };

  const parsedResults = useMemo(() => {
    if (!results || typeof results !== 'object') return {};
    return Object.fromEntries(
      Object.entries(results).filter(([, image]) => {
        return !(image[0].display_type && image[0].display_type == OutputDisplayType.Hidden);
      }),
    );
  }, [results]);

  return (
    <div id="results-tab" className="navbar-scroll-offset results-container">
      <LoadingSpinner loading={loading}></LoadingSpinner>
      {parsedResults && typeof parsedResults === 'object' && !loading && (
        <>
          <div>
            {numResults == 0 && !loading && (
              <>
                <br />
                <AlertBanner severity={Severity.Info}>
                  <h3>No Tool Results Available</h3>
                </AlertBanner>
              </>
            )}
            {Object.keys(parsedResults)
              .sort()
              .map((image) => (
                <ToolResult
                  key={image}
                  header={image}
                  tool={image}
                  sha256={sha256}
                  updateInView={updateInView}
                  updateURLSection={updateURLSection}
                  results={parsedResults[image]}
                />
              ))}
          </div>
          {Object.keys(parsedResults).length > 0 && (
            <div className="results-toc-col">
              <ResultsTableOfContents parsedResults={parsedResults} inViewElements={inViewElements} />
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Results;
