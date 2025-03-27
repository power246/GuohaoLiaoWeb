package student;
/**
 * This class is the main program.
 * Calculate the winning rate for each type of AI.
 *
 * @author Paul Liao
 */
public class Tournament {
    /**
     * This is the main method which makes use of addNum method.
     *
     * @param args Unused.
     */
    public static void main(String[] args) {
        AI r = new AI();
        SmallestCardAI s = new SmallestCardAI();
        BiggestCardAI b = new BiggestCardAI();
        AI[] playList = new AI[]{r, s, b};
        for (AI p1 : playList) {
            for (AI p2 : playList) {
                UnoWarMatch trail = new UnoWarMatch(p1, p2);
                System.out.println(p1 + " vs. " + p2 + " winRate: " + trail.winRate(1000));
            }
        }
    }
}
