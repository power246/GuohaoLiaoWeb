package student;
import java.util.Random;

/**
 * This class is for creating the deck.
 *
 * @author Paul Liao
 */

public class Deck {

    /** The cards that deck contained. */
    private Card[] deck = new Card[52];
    /** The remaining cards numbers. */
    private int remaining = 52; //

    /**
     * Constructor that create the new deck.
     */
    public Deck() {
        // Create a new deck.
        int num = 0;
        for (int i = 1; i <= 4; i++) {
            for (int ii = 1; ii <= 13; ii++) {
                deck[num] = new Card(ii, i);
                num++;
            }
        }
        shuffle();
    }

    /**
     * Use Fisher-Yates-Knuth algorithm to shuffle the deck.
     */
    public void shuffle() {
        Random random = new Random();
        for (int i = deck.length - 1; i >= 1; i--) {
            int ranInd = random.nextInt(i + 1);
            Card flag = deck[i];
            deck[i] = deck[ranInd];
            deck[ranInd] = flag;
        }
    }

    /**
     * To draw and return the next card.
     *
     * @return Card This returns the card been drawn.
     */
    public Card draw() {
        Card card;
        if (isEmpty()) {
            shuffle();
            remaining = 52;
        }
        card = deck[remaining - 1];
        remaining--;
        return card;
    }

    /**
     * It gets the remaining card number.
     *
     * @return int This returns the remaining card number.
     */
    public int cardsRemaining() {
        return remaining;
    }

    /**
     * To check if the deck is empty.
     *
     * @return boolean This returns true if the deck is empty.
     */
    public boolean isEmpty() {
        return remaining == 0;
    }
}
