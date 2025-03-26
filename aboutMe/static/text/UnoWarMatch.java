package student;

/**
 * This class is for game setting and playing.
 *
 * @author Paul Liao
 */
public class UnoWarMatch {


    /** The first AI player. */
    private AI ai1 = null;
    /** The second AI player. */
    private AI ai2 = null;

    /**
     * Constructor that takes two AIs to play.
     *
     * @param a This is the first AI.
     * @param b This is the second AI.
     */
    public UnoWarMatch(AI a, AI b) {
        ai1 = a;
        ai2 = b;
    }

    /**
     * Game play method. If player 1 (ai1) wins, return true;
     * if player 2 (ai2) wins, return false.
     *
     * @return boolean This returns the match result.
     */
    public boolean playGame() {
        // Game play method. If ai1 wins, return true;
        // if ai2 wins, return false.
        AI p1 = ai1;
        AI p2 = ai2;
        Deck deck = new Deck();
        Hand h1 = new Hand(deck, 5);
        Hand h2 = new Hand(deck, 5);
        CardPile pile = new CardPile(deck.draw());
        Card card = null;
        int p1Win = 0;
        int p2Win = 0;
        while (p1Win < 10 && p2Win < 10) {
            card = p1.getPlay(h1, pile);
            if (card != null) {
                h1.remove(card);
                pile.play(card);
            } else {
                p2Win++;
                pile = new CardPile(deck.draw());
            }
            card = p2.getPlay(h2, pile);
            if (card != null) {
                h2.remove(card);
                pile.play(card);
            } else {
                p1Win++;
                pile = new CardPile(deck.draw());
            }
        }
        return p1Win == 10;
    }

    /**
     * The AIs play each other nTrials times, and report the percent of
     * times AI1 beat AI2 as a double.
     *
     * @param nTrials This represents the number of trials.
     * @return double This returns the winning rate.
     */
    public double winRate(int nTrials) {
        double p1Times = 0.0;
        for (int i = 0; i < nTrials; i++) {
            if (playGame()) {
                p1Times += 1;
            }
        }
        return p1Times / nTrials;
    }
}
